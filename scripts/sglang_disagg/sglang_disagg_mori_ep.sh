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
xP="${xP:-1}"
yD="${yD:-1}"
IPADDRS="${IPADDRS:-localhost}"
BARRIER_PORT="${BARRIER_PORT:-4342}"

IB_DEVICES=${RDMA_IFNAME:-"mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9"}

export GLOO_SOCKET_IF_NAME=${IFNAME:-eth0}
export NCCL_SOCKET_IF_NAME=${IFNAME:-eth0}

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================

pip install py-spy
pip install --ignore-installed --force-reinstall flask

host_ip=$(hostname -I | awk '{print $1}')
host_name=$(hostname)

# =============================================================================
# Model-Specific Configuration Maps
# =============================================================================

declare -A MODEL_PREFILL_CONFIGS=(
    ["DeepSeek-R1"]="--moe-a2a-backend mori --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head --attention-backend aiter --watchdog-timeout 1000000 --mem-fraction-static 0.8 --max-running-requests 8 --cuda-graph-bs $(seq 1 3) --disable-radix-cache"
)

declare -A MODEL_DECODE_CONFIGS=(
    ["DeepSeek-R1"]="--moe-a2a-backend mori --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head --attention-backend aiter --watchdog-timeout 1000000 --mem-fraction-static 0.8 --max-running-requests 8192 --cuda-graph-bs $(seq 1 64) --disable-radix-cache"
)

# =============================================================================
# Configuration Selection Functions
# =============================================================================

get_model_config() {
    local mode="$1"
    local model_name="$2"
    
    if [[ "$mode" == "prefill" ]]; then
        if [[ -n "${MODEL_PREFILL_CONFIGS[$model_name]}" ]]; then
            echo "${MODEL_PREFILL_CONFIGS[$model_name]}"
        else
            echo "ERROR: No prefill model configuration found for $model_name"
            exit 1
        fi
    elif [[ "$mode" == "decode" ]]; then
        if [[ -n "${MODEL_DECODE_CONFIGS[$model_name]}" ]]; then
            echo "${MODEL_DECODE_CONFIGS[$model_name]}"
        else
            echo "ERROR: No decode model configuration found for $model_name"
            exit 1
        fi
    fi
}

if [[ -z "$MODEL_NAME" ]]; then
    echo "ERROR: MODEL_NAME not set, exiting"
    exit 1
else
    PREFILL_MODEL_CONFIG=$(get_model_config "prefill" "$MODEL_NAME")
    DECODE_MODEL_CONFIG=$(get_model_config "decode" "$MODEL_NAME")
    echo "Using model-specific configuration for: $MODEL_NAME"
fi

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
# Cluster Topology Configuration
# =============================================================================

IP_FIRST_PREFILL=$(echo "$IPADDRS" | awk -F',' '{print $2}')
IP_FIRST_DECODE=$(echo "$IPADDRS" | awk -F',' -v pos="$xP" '{print $(pos+2)}')

IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_ARGS=""
DECODE_ARGS=""

# Loop through for `--prefill` (IPs from index 0 to N-1)
for ((i=1; i<=$xP && i<${#IP_ARRAY[@]}; i++)); do
    PREFILL_ARGS+="http://${IP_ARRAY[$i]}:3000 "
done

# Loop through for `--decode` (IPs from N onward)
for ((i=$xP+1; i<${#IP_ARRAY[@]}; i++)); do
    DECODE_ARGS+="http://${IP_ARRAY[$i]}:3000 "
done

# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================

cd /sgl-workspace/sglang

if [ "$NODE_RANK" -eq 0 ]; then
    echo "NODE INFO ======================================="
    echo "================================================"
    echo "Node List : ${SLURM_JOB_NODELIST}"
    echo "Node IPs : ${IPADDRS}"
    echo "Model Name : ${MODEL_NAME:-'Not specified'}"
    echo "================================================"

    echo "CLUSTER INFO ===================================="
    echo "================================================"
    echo "${host_name}:${host_ip} is Proxy Node"
    echo "${PREFILL_ARGS} are Proxy's Prefill"
    echo "${DECODE_ARGS} are Proxy's Decode"
    echo "================================================"

    TIMEOUT_SECONDS=4000
    SLEEP_SECONDS=10
    SEARCH_SIGNAL="The server is fired up and ready to roll!"
    SECONDS=0
    sleep 20;
    for ((i=1; i<2; i++)); do
         LOG_FILE=/run_logs/${SLURM_JOB_ID}/prefill_NODE${i}.log
         #wait until prefill nodes get ready
         until grep -q "${SEARCH_SIGNAL}" "${LOG_FILE}"; do
             if [ $SECONDS -ge $TIMEOUT_SECONDS ]; then
                 echo "Awaited ${SECONDS} seconds. Timeout reached. Signal not found in prefill ${i} file" \
     			| tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null
             	
             fi
             sleep $SLEEP_SECONDS
     	SECONDS=$(( SECONDS + SLEEP_SECONDS))
      done      
     done

    for ((i=$xP+1; i<$xP+2; i++)); do
         LOG_FILE=/run_logs/${SLURM_JOB_ID}/decode_NODE${i}.log
         #wait until decode nodes get ready         
         until grep -q "${SEARCH_SIGNAL}" "${LOG_FILE}"; do
            if [ $SECONDS -ge $TIMEOUT_SECONDS ]; then
               echo "Awaited ${SECONDS} seconds. Timeout reached. Signal not found in decode ${i} file" \
     		       | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null
            fi
            sleep $SLEEP_SECONDS
            SECONDS=$(( SECONDS + SLEEP_SECONDS))
         done
     done

    sleep 10

    python -m sglang_router.launch_router \
    	--pd-disaggregation \
         --prefill http://${IP_FIRST_PREFILL}:3000 \
         --decode  http://${IP_FIRST_DECODE}:3000 \
         --host 0.0.0.0 \
         --port 2322 \
         2>&1 | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &
    
    proxy_pid=$!
    
    echo "Waiting for all prefill and decode servers to be up . . ."
    #python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
    #    --node-ips ${IPADDRS} \
    #   --node-ports 3000
    sleep 60;

    echo "Proxy Server Ready for benchmarking on ${host_name}:${host_ip}"

    sleep 10
    cd /opt/mooncake-cookbook
    bash /opt/mooncake-cookbook/benchmark_xPyD.sh

    echo "Killing the proxy server"
    kill $proxy_pid

elif  [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -le "$xP" ]; then

    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-'default'})"

    export GLOO_SOCKET_IF_NAME=${IFNAME:-eth0}
    export NCCL_SOCKET_IF_NAME=${IFNAME:-eth0}
    export SGLANG_USE_AITER=1
    export SGLANG_MORI_FP8_DISP=True
    export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

#["DeepSeek-R1"]="--moe-a2a-backend mori --moe-dense-tp-size 1 --enable-dp-lm-head --attention-backend aiter --watchdog-timeout 1000000 --mem-fraction-static 0.8 --max-running-requests 8 --cuda-graph-bs $(seq 1 3) --disable-radix-cache

    python3 -m sglang.launch_server \
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
        --node-rank $((${NODE_RANK}-1)) \
        --tp-size $((${xP}*8)) \
        --dp-size $((${xP}*8)) \
        --ep-size $((${xP}*8)) \
        --decode-log-interval 1 \
        $PREFILL_MODEL_CONFIG \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &

    prefill_pid=$!

    echo "Waiting for proxy server to be up..."
    python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports 2322

    echo "Waiting untill proxy server closes..."
    python $MOONCAKE_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port 2322

    echo "Killing the prefill server"
    kill $prefill_pid
else
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-'default'})"
    #echo "Using decode config: $DECODE_MODEL_CONFIG"

    export GLOO_SOCKET_IF_NAME=${IFNAME:-eth0}
    export NCCL_SOCKET_IF_NAME=${IFNAME:-eth0}
    export SGLANG_USE_AITER=1
    export SGLANG_MORI_FP8_DISP=True
    export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

    python3 -m sglang.launch_server \
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
        --node-rank $((${NODE_RANK}-${xP}-1)) \
        --tp-size $((${yD}*8)) \
        --dp-size $((${yD}*8)) \
        --ep-size $((${yD}*8)) \
        --decode-log-interval 1 \
        $DECODE_MODEL_CONFIG \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/decode_NODE${NODE_RANK}.log >/dev/null &

    decode_pid=$!

    echo "Waiting for proxy server to be up..."
    python $MOONCAKE_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports 2322

    echo "Waiting untill proxy server closes..."
    python $MOONCAKE_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port 2322

    echo "Killing the decode server"
    kill $decode_pid
fi

echo "Script completed successfully"
exit 0
