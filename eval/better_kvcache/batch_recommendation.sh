
for document_length in 5000; do
    python benchmark_recommendation.py --document-length $document_length
    python benchmark_recommendation.py --document-length $document_length --prefill-only
done
