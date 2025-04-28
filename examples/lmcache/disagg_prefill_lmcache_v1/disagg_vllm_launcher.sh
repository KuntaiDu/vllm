#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <prefiller | decoder> [model]"
    exit 1
fi

if [[ $# -eq 1 ]]; then
    echo "Using default model: meta-llama/Llama-3.1-8B-Instruct"
    MODEL="meta-llama/Llama-3.1-8B-Instruct"
else
    echo "Using model: $2"
    MODEL=$2
fi

if [[ -n "${NSYS_PROFILE:-}" ]]; then
    echo "NSYS_PROFILE is set, checking if nsys exists..."
    if command -v nsys >/dev/null 2>&1; then
        echo "nsys is installed"
    else
        echo "nsys is not installed"
        exit 1
    fi
fi


# set prefiller / decoder specific variables.
if [[ $1 == "prefiller" ]]; then
    CONFIG_FILE="$SCRIPT_DIR/configs/lmcache-prefiller-config.yaml"
    PORT=8100
    GPU=0
    ROLE="kv_producer"
    RPC_PORT="producer1"
    OUTPUT="prefiller"
elif [[ $1 == "decoder" ]]; then
    CONFIG_FILE="$SCRIPT_DIR/configs/lmcache-decoder-config.yaml"
    PORT=8200
    GPU=1
    ROLE="kv_consumer"
    RPC_PORT="consumer1"
    OUTPUT="decoder"
else
    echo "Invalid role: $1"
    echo "Should be either prefiller or decoder"
    exit 1
fi

export UCX_TLS=cuda_ipc,cuda_copy,tcp
export LMCACHE_CONFIG_FILE="$CONFIG_FILE"
export LMCACHE_USE_EXPERIMENTAL=True
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=$GPU

# Base vllm serve command
VLLM_CMD=(
    vllm serve "$MODEL"
    --port "$PORT"
    --disable-log-requests
    --enforce-eager
    --kv-transfer-config
    "{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"$ROLE\",\"kv_connector_extra_config\": {\"discard_partial_chunks\": false, \"lmcache_rpc_port\": \"$RPC_PORT\"}}"
)

# If NSYS_PROFILE is set, wrap with nsys
if [[ -n "${NSYS_PROFILE:-}" ]]; then
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --output="$OUTPUT" \
        --force-overwrite true \
        "${VLLM_CMD[@]}"
else
    "${VLLM_CMD[@]}"
fi
