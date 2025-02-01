import vllm
import torch

torch.cuda.set_per_process_memory_fraction(0.6, device=None)

clen = 54000

samp = vllm.SamplingParams(max_tokens=1)
llm = vllm.LLM(model="meta-llama/Llama-3.1-8B-Instruct",
               enable_chunked_prefill=True,
               enforce_eager=True,
               max_model_len=clen + 50,
               gpu_memory_utilization=0.59,
               block_size=16,
               )

output = llm.generate("Hi" * clen, samp)[0]

print(output.outputs)
