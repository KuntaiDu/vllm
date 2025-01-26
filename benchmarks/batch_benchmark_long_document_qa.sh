
# for bs in 16 64 256 1024; do
for bs in 16 64 256 1024; do
    python benchmark_long_document_qa.py \
        --block-allocator CpuOffloadingBlockAllocator \
        --cpu-memory-gb 96 \
        --gpu-memory-utilization 0.6 \
        --num-documents 8 \
        --document-length 20481 \
        --enable-prefix-caching \
        --profile-swap-blocks \
        --block-size $bs \
        --profile-forward

    python benchmark_long_document_qa.py \
        --block-allocator CpuOffloadingBlockAllocator \
        --cpu-memory-gb 96 \
        --gpu-memory-utilization 0.6 \
        --num-documents 8 \
        --document-length 20481 \
        --enable-prefix-caching \
        --profile-swap-blocks \
        --block-size $bs \
        --profile-forward \
        --non-consecutive-alloc
done