
for document_length in 1000; do
    python benchmark_recommendation.py --document-length $document_length --num-documents 500
    python benchmark_recommendation.py --document-length $document_length --prefill-only --num-documents 500
done
