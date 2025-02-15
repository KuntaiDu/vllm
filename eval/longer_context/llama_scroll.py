import vllm
import torch
import os

os.environ['PREFILL_ONLY'] = '1'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['CHUNK_SIZE'] = "2048"

torch.cuda.set_per_process_memory_fraction(0.59, device=None)

# MLEN = 211000
MLEN = 230000

samp = vllm.SamplingParams(max_tokens=1)
llm = vllm.LLM(model="meta-llama/Llama-3.1-8B-Instruct",
               enforce_eager=True,
               max_model_len=MLEN + 100,
               gpu_memory_utilization=0.59,
               block_size=16,
               enable_chunked_prefill=False,
               enable_prefix_caching=True)

output = llm.generate("Hi" * MLEN, samp)[0]

print(output.outputs)

output = llm.generate("Hi" * MLEN, samp)[0]

print(output.outputs)
