
for gpu_util in 0.53 0.98; do
    python benchmark_long_document_qa.py \
        --block-allocator CpuOffloadingBlockAllocator \
        --cpu-memory-gb 96 \
        --gpu-memory-utilization 0.98 \
        --num-documents 20 \
        --document-length 20481 \
        --enable-prefix-caching \
        --profile-swap-blocks \
        --profile-forward \
        --output-len 1
done