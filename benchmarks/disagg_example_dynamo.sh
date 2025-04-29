#!/bin/bash

echo "Warning: LMCache disaggregated prefill support for vLLM v1 is experimental and subject to change."


PIDS=()

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

check_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. Please set it to your Hugging Face token."
        exit 1
    fi
    if [[ "$HF_TOKEN" != hf_* ]]; then
        echo "HF_TOKEN is not a valid Hugging Face token. Please set it to your Hugging Face token."
        exit 1
    fi
    echo "HF_TOKEN is set and valid."
}

check_num_gpus() {
    # can you check if the number of GPUs are >=2 via nvidia-smi?
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    python -c "import $1" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        if [ "$1" == "nixl" ]; then
            echo "$1 is not installed. Please refer to https://github.com/ai-dynamo/nixl for installation."
        else
            echo "$1 is not installed. Please install it via pip install $1."
        fi
        exit 1
    else
        echo "$1 is installed."
    fi
}

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM        # prevent re-entrancy
    kill -- -$$            # negative PID  ==  “this whole process-group”
    wait                   # reap children so we don't leave zombies
    exit 0
}

wait_for_server() {
  local port=$1
  local timeout_seconds=1200
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}/v1/completions" > /dev/null; then
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server"
      return 1
    fi

    sleep 1
  done
}


main() {
    check_hf_token
    check_num_gpus
    ensure_python_library_installed nixl
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "Launching prefiller, decoder and proxy..."
    echo "Please check prefiller.log, decoder.log and proxy.log for logs."

    bash disagg_dynamo_launcher.sh prefiller \
        > >(tee prefiller.log) 2>&1 &
    prefiller_pid=$!
    PIDS+=($prefiller_pid)

    bash disagg_dynamo_launcher.sh decoder  \
        > >(tee decoder.log)  2>&1 &
    decoder_pid=$!
    PIDS+=($decoder_pid)

    bash disagg_dynamo_launcher.sh prefiller2  \
        > >(tee decoder.log)  2>&1 &
    decoder_pid=$!
    PIDS+=($decoder_pid)

    wait_for_server 8080
    wait_for_server 8090
    wait_for_server 8100

    # Establish nixl conn
    curl -kvvv -XPOST http://127.0.0.1:8080/add_remote_prefill_eps  -H "Content-Type: application/json" -d '{"endpoints":["http://127.0.0.1:8090"]}'
    curl -kvvv -XPOST http://127.0.0.1:8080/add_remote_prefill_eps  -H "Content-Type: application/json" -d '{"endpoints":["http://127.0.0.1:8100"]}'

    echo "All servers are up. Starting benchmark..."


    # begin benchmark
    # python benchmark_serving.py --port 8080 --seed $(date +%s) \
    #     --model meta-llama/Llama-3.1-8B-Instruct \
    #     --dataset-name random --random-input-len 7500 --random-output-len 200 \
    #     --num-prompts 200 --burstiness 100 --request-rate 3.6 \
    #     --backend openai-chat --endpoint /v1/chat/completions | tee benchmark.log

    # 1P1D dynamo
    # python3 benchmark_serving.py --port 8080 --seed $(date +%s) \
    #     --model meta-llama/Llama-3.1-8B-Instruct \
    #     --dataset-name random --random-input-len 8000 --random-output-len 200 \
    #     --num-prompts 200 --burstiness 100 --request-rate 3.6 --metric-percentiles 95 --backend openai-chat --endpoint /v1/chat/completions --ignore-eos | tee benchmark.log

    # 2P1D dynamo
    python3 benchmark_serving.py --port 8080 --seed $(date +%s)         --model meta-llama/Llama-3.1-8B-Instruct         --dataset-name random --random-input-len 10000 --random-output-len 100         --num-prompts 250 --burstiness 100 --request-rate 5.5 --metric-percentiles 95 --backend openai-chat --endpoint /v1/chat/completions --ignore-eos | tee benchmark.log

    echo "Benchmarking done. Cleaning up..."

    cleanup

}

main