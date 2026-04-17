#!/bin/bash
# MoRI EP PD entrypoint (used when RUN_MORI=1 in run_xPyD_models.slurm).
# Customize for MoRI expert-parallel + disaggregated launch; until then this
# delegates to the standard Mooncake PD launcher.

# =============================================================================
# Environment Configuration
# =============================================================================

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_PATH=$MODEL_PATH
MODEL_NAME="${MODEL_NAME:-}"
PARALLEL_MODE="${PARALLEL_MODE:-dp}" # supported: dp, tp
xP="${xP:-1}"
yD="${yD:-1}"
DP_MODE="${DP_MODE:-0}"
IPADDRS="${IPADDRS:-localhost}"
BARRIER_PORT="${BARRIER_PORT:-4342}"
IB_DEVICES=${IB_DEVICES:-"mlx5_0"}

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================

pip install py-spy
pip install --ignore-installed --force-reinstall flask
pip install pyyaml

host_ip=$(ip route get 1.1.1.1 | awk '/src/ {print $7}')
host_name=$(hostname)

if [[ "$PARALLEL_MODE" != "dp" && "$PARALLEL_MODE" != "tp" ]]; then
    echo "ERROR: PARALLEL_MODE must be 'dp' or 'tp' (got: ${PARALLEL_MODE})"
    exit 1
fi

# =============================================================================
# Parallelism Settings
# =============================================================================

# Parallelism from node counts (xP prefill, yD decode). --tp-size always;
# --dp-size and --ep-size only when DP_MODE=1 (same total degree as Nnodes×GPUS_PER_NODE).
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

PREFILL_TP_SIZE=$((xP * GPUS_PER_NODE))
PREFILL_EP_SIZE=$((xP * GPUS_PER_NODE))

DECODE_TP_SIZE=$((yD * GPUS_PER_NODE))
DECODE_EP_SIZE=$((yD * GPUS_PER_NODE))

if [[ "$DP_MODE" == "1" ]]; then
    PREFILL_DP_SIZE=$((xP * GPUS_PER_NODE))
    DECODE_DP_SIZE=$((yD * GPUS_PER_NODE))
    export PREFILL_DP_SIZE DECODE_DP_SIZE
else
    unset PREFILL_DP_SIZE DECODE_DP_SIZE 2>/dev/null || true
fi
export PREFILL_TP_SIZE PREFILL_EP_SIZE DECODE_TP_SIZE DECODE_EP_SIZE

# =============================================================================
# Model-Specific Configuration from YAML
# =============================================================================

if [[ -z "$MODEL_NAME" ]]; then
    echo "ERROR: MODEL_NAME not set, exiting"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_YAML="${MODELS_YAML:-${SCRIPT_DIR}/models.yaml}"

if [[ ! -f "$MODELS_YAML" ]]; then
    echo "ERROR: models.yaml not found at $MODELS_YAML"
    exit 1
fi

export MODELS_YAML MODEL_NAME PARALLEL_MODE
eval "$(python3 - <<'PY'
import os
import shlex
import sys
import yaml

config_path = os.environ["MODELS_YAML"]
model_name = os.environ["MODEL_NAME"]
mode = os.environ["PARALLEL_MODE"]

with open(config_path, "r", encoding="utf-8") as f:
    models = yaml.safe_load(f) or {}

if model_name not in models:
    print(f'echo "ERROR: Model {model_name} not found in {config_path}"; exit 1')
    sys.exit(0)

cfg = models[model_name] or {}
prefill = cfg.get("prefill", {}) or {}
decode = cfg.get("decode", {}) or {}


def q(v):
    return shlex.quote(str(v if v is not None else ""))


exports = {
    "MODEL_BASE_FLAGS": cfg.get("base_flags", ""),
    "MODEL_MODE_FLAGS": cfg.get(f"{mode}_flags", ""),
    "MODEL_PREFILL_FLAGS": prefill.get(mode, ""),
    "MODEL_DECODE_FLAGS": decode.get(mode, ""),
}

for key, value in exports.items():
    print(f"{key}={q(value)}")
PY
)"

PREFILL_MODEL_CONFIG="${MODEL_BASE_FLAGS} ${MODEL_MODE_FLAGS} ${MODEL_PREFILL_FLAGS}"
DECODE_MODEL_CONFIG="${MODEL_BASE_FLAGS} ${MODEL_MODE_FLAGS} ${MODEL_DECODE_FLAGS}"
echo "Using model-specific configuration for: $MODEL_NAME (mode=${PARALLEL_MODE})"

export PREFILL_MODEL_CONFIG DECODE_MODEL_CONFIG

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/mori_ep_env.sh"

# =============================================================================
# Cluster Topology (dist-init endpoints)
# =============================================================================

IP_FIRST_PREFILL=$(echo "$IPADDRS" | awk -F',' '{print $2}')
IP_FIRST_DECODE=$(echo "$IPADDRS" | awk -F',' -v pos="$xP" '{print $(pos+2)}')

IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_ARGS=""
DECODE_ARGS=""

# Router (DP_MODE=0): one --prefill / --decode URL per worker (see sglang_disagg_server.sh).
for ((i=1; i<=$xP && i<${#IP_ARRAY[@]}; i++)); do
    PREFILL_ARGS+=" --prefill http://${IP_ARRAY[$i]}:3000"
done

for ((i=$xP+1; i<${#IP_ARRAY[@]}; i++)); do
    DECODE_ARGS+=" --decode http://${IP_ARRAY[$i]}:3000"
done

echo "PREFILL_ARGS: $PREFILL_ARGS"
echo "DECODE_ARGS: $DECODE_ARGS"


# =============================================================================
# Container Synchronization
# =============================================================================

echo "Waiting at the container creation barrier on $host_name"
python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
    --local-ip ${host_ip} \
    --local-port ${BARRIER_PORT} \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports ${BARRIER_PORT}


# =============================================================================
# Prepared sglang launch commands
# =============================================================================
# NODE_RANK 0: sglang_router (DP_MODE=0: all PREFILL_ARGS/DECODE_ARGS; DP_MODE=1: first prefill/decode only).
# NODE_RANK 1..xP: prefill workers (PREFILL_NODE_RANK = NODE_RANK - 1).
# NODE_RANK xP+1 .. xP+yD: decode workers (DECODE_NODE_RANK = NODE_RANK - xP - 1).
# After setup_sglang_worker_env (see sglang_disagg_mori_ep.sh), run eval "$PREFILL_CMD" or eval "$DECODE_CMD".

cd /sgl-workspace/sglang || {
    echo "ERROR: cd /sgl-workspace/sglang failed"
    exit 1
}

unset PREFILL_CMD DECODE_CMD ROUTER_CMD 2>/dev/null || true

setup_sglang_worker_env() {
    export GLOO_SOCKET_IF_NAME="${GLOO_SOCKET_IF_NAME:-${IFNAME:-eth0}}"
    export NCCL_SOCKET_IF_NAME="${NCCL_SOCKET_IF_NAME:-${IFNAME:-eth0}}"
    export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
    export SGLANG_MORI_FP8_DISP="${SGLANG_MORI_FP8_DISP:-True}"
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-1200}"
}

if [[ "$NODE_RANK" -eq 0 ]]; then
    echo "${host_name}:${host_ip} is Router / proxy (NODE_RANK=0)"

    if [[ "$DP_MODE" == "1" ]]; then
        ROUTER_CMD="python3 -m sglang_router.launch_router \
--pd-disaggregation \
--prefill http://${IP_FIRST_PREFILL}:3000 \
--decode http://${IP_FIRST_DECODE}:3000 \
--host ${host_ip} \
--port 2322"
    else
        ROUTER_CMD="python3 -m sglang_router.launch_router \
--pd-disaggregation \
${PREFILL_ARGS} \
${DECODE_ARGS} \
--host ${host_ip} \
--port 2322"
    fi
    export ROUTER_CMD
    mkdir -p "/run_logs/${SLURM_JOB_ID:-0}"
    set -x
    eval "$ROUTER_CMD" 2>&1 | tee "/run_logs/${SLURM_JOB_ID:-0}/proxy_NODE${NODE_RANK}.log" >/dev/null &
    set +x
    proxy_pid=$!
    echo "Router (sglang_router) started pid=${proxy_pid} (DP_MODE=${DP_MODE})"

elif [[ "$NODE_RANK" -ge 1 && "$NODE_RANK" -le "$xP" ]]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-default})"
    PREFILL_NODE_RANK=$((NODE_RANK - 1))
    setup_sglang_worker_env
    export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${MORI_MAX_DISPATCH_TOKENS_PREFILL}"

    PREFILL_CMD="python3 -m sglang.launch_server \
--model-path ${MODEL_PATH} \
--disaggregation-mode prefill \
--disaggregation-transfer-backend mori \
--load-balance-method round_robin \
--disaggregation-ib-device ${IB_DEVICES} \
--host ${host_ip} \
--port 3000 \
--trust-remote-code \
--dist-init-addr ${IP_FIRST_PREFILL}:5757 \
--nnodes ${xP} \
--node-rank ${PREFILL_NODE_RANK} \
--tp-size ${PREFILL_TP_SIZE}"
    if [[ "$DP_MODE" == "1" ]]; then
        PREFILL_CMD+=" \
--dp-size ${PREFILL_DP_SIZE} \
--ep-size ${PREFILL_EP_SIZE}"
    fi
    PREFILL_CMD+=" \
--decode-log-interval 1 \
${PREFILL_MODEL_CONFIG} \
--log-level-http warning"
    export PREFILL_CMD PREFILL_NODE_RANK

    set -x
    eval "$PREFILL_CMD" 2>&1 | tee "/run_logs/${SLURM_JOB_ID:-0}/prefill_NODE${NODE_RANK}.log" >/dev/null &
    set +x
    prefill_pid=$!

    echo "Waiting for proxy server to be up..."
    python "$MOONCAKE_COOKBOOK_PATH/socket_barrier.py" \
        --node-ips "${MASTER_ADDR}" \
        --node-ports 2322

    echo "Waiting until proxy server closes..."
    python "$MOONCAKE_COOKBOOK_PATH/socket_wait.py" \
        --remote-ip "${MASTER_ADDR}" \
        --remote-port 2322

    echo "Killing the prefill server"
    kill "${prefill_pid}"

elif [[ "$NODE_RANK" -ge $((xP + 1)) && "$NODE_RANK" -le $((xP + yD)) ]]; then
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-default})"
    DECODE_NODE_RANK=$((NODE_RANK - xP - 1))
    setup_sglang_worker_env
    export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${MORI_MAX_DISPATCH_TOKENS_DECODE}"
    if [[ "$DP_MODE" == "1" ]]; then
        export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD="${SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD:-$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))}"
    fi

    DECODE_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_DECODE} python3 -m sglang.launch_server \
--model-path ${MODEL_PATH} \
--disaggregation-mode decode \
--disaggregation-transfer-backend mori \
--load-balance-method round_robin \
--prefill-round-robin-balance \
--disaggregation-ib-device ${IB_DEVICES} \
--host ${host_ip} \
--port 3000 \
--trust-remote-code \
--dist-init-addr ${IP_FIRST_DECODE}:5757 \
--nnodes ${yD} \
--node-rank ${DECODE_NODE_RANK} \
--tp-size ${DECODE_TP_SIZE}"
    if [[ "$DP_MODE" == "1" ]]; then
        DECODE_CMD+=" \
--dp-size ${DECODE_DP_SIZE} \
--ep-size ${DECODE_EP_SIZE}"
    fi
    DECODE_CMD+=" \
--decode-log-interval 1 \
${DECODE_MODEL_CONFIG} \
--log-level-http warning"
    export DECODE_CMD DECODE_NODE_RANK

    set -x
    eval "$DECODE_CMD" 2>&1 | tee "/run_logs/${SLURM_JOB_ID:-0}/decode_NODE${NODE_RANK}.log" >/dev/null &
    set +x
    decode_pid=$!

    echo "Waiting for proxy server to be up..."
    python "$MOONCAKE_COOKBOOK_PATH/socket_barrier.py" \
        --node-ips "${MASTER_ADDR}" \
        --node-ports 2322

    echo "Waiting until proxy server closes..."
    python "$MOONCAKE_COOKBOOK_PATH/socket_wait.py" \
        --remote-ip "${MASTER_ADDR}" \
        --remote-port 2322

    echo "Killing the decode server"
    kill "${decode_pid}"

else
    echo "ERROR: NODE_RANK=${NODE_RANK} out of range (expected 0..$((xP + yD))) for xP=${xP} yD=${yD}" >&2
    exit 1
fi

echo "Script completed successfully"
exit 0

