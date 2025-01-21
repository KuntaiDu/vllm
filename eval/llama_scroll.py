import vllm
import torch
import os

os.environ['PREFILL_ONLY'] = '1'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

torch.cuda.set_per_process_memory_fraction(0.45, device=None)

MLEN = 190000

samp = vllm.SamplingParams(max_tokens=1)
llm = vllm.LLM(model="meta-llama/Llama-3.1-8B-Instruct",
               enable_chunked_prefill=True,
               enforce_eager=True,
               max_model_len=MLEN + 100,
               gpu_memory_utilization=0.44,
               block_size=16,
               max_num_batched_tokens=4096)

output = llm.generate("Hi" * MLEN, samp)[0]

print(output.outputs)
