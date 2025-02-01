
import vllm
import torch
import os


prefill_only = True

os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

if prefill_only:
    os.environ['PREFILL_ONLY'] = '1'
    os.environ['CHUNK_SIZE'] = "2048"

torch.cuda.set_per_process_memory_fraction(0.6, device=None)

if __name__ == "__main__":
    MLEN = 190000

    samp = vllm.SamplingParams(max_tokens=1)
    llm = vllm.LLM(model="meta-llama/Llama-3.1-8B-Instruct",
                enforce_eager=True,
                max_model_len=MLEN + 100,
                gpu_memory_utilization=0.59,
                block_size=16,
                enable_chunked_prefill=not prefill_only,
                tensor_parallel_size=1 if prefill_only else 2,
                )

    import time
    
    st = time.time()
    torch.cuda.synchronize()
    for i in range(3 * (1 if prefill_only else 2)):
        output = llm.generate("Hi" * MLEN, samp)[0]
    torch.cuda.synchronize()
    ed = time.time()

    print(ed - st)
