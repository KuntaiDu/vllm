#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

get_blocksize() {
    if [ -z "${BLOCK_SIZE}" ]; then
        echo "128"
    else 
        echo "${BLOCK_SIZE}"
    fi
}

BLOCK_SIZE=$(get_blocksize)

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
    PORT=8090
    GPU=0
    OUTPUT=prefiller
elif [[ $1 == "decoder" ]]; then
    PORT=8080
    GPU=1
    OUTPUT=decoder
else
    echo "Invalid role: $1"
    echo "Should be either prefiller or decoder"
    exit 1
fi

export UCX_TLS=cuda_ipc,cuda_copy,tcp
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=$GPU

# Base vllm serve command
VLLM_CMD=(
    vllm serve "$MODEL"
    --port "$PORT"
    --swap-space 0
    --block-size $BLOCK_SIZE
    --trust-remote-code
    --disable-log-requests
    --enforce-eager
    --enable-chunked-prefill false
    --kv-transfer-config
    "{\"kv_connector\":\"DynamoNixlConnector\"}"
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
