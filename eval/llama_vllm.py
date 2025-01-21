import vllm
import torch

torch.cuda.set_per_process_memory_fraction(0.45, device=None)

samp = vllm.SamplingParams(max_tokens=1)
llm = vllm.LLM(model="meta-llama/Llama-3.1-8B-Instruct",
               enable_chunked_prefill=True,
               enforce_eager=True,
               max_model_len=7050,
               gpu_memory_utilization=0.44,
               block_size=16)

output = llm.generate("Hi" * 7000, samp)[0]

print(output.outputs)
