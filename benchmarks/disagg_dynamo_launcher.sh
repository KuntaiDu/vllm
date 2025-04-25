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


if [[ $1 == "prefiller" ]]; then

    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=0 \
        vllm serve $MODEL \
        --port 8090 \
        --swap-space 0 \
        --block-size $BLOCK_SIZE \
        --trust-remote-code \
        --enforce-eager \
        --kv-transfer-config '{"kv_connector":"DynamoNixlConnector", "kv_connector_extra_config":{"skip_sampling":true}}' \
        --enable-chunked-prefill false \
        --disable-log-requests 

elif [[ $1 == "decoder" ]]; then

    UCX_TLS=cuda_ipc,cuda_copy,tcp \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=1 \
        vllm serve $MODEL \
        --port 8080 \
        --swap-space 0 \
        --block-size $BLOCK_SIZE \
        --trust-remote-code \
        --kv-transfer-config '{"kv_connector":"DynamoNixlConnector"}' \
        --enable-chunked-prefill false \
        --disable-log-requests \
        --enforce-eager 


else
    echo "Invalid role: $1"
    echo "Should be either prefill, decode"
    exit 1
fi
