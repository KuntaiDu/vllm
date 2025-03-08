import os
import cProfile
import pstats
import random

# os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
# os.environ['CHUNK_SIZE'] = "2048"
# os.environ['VLLM_USE_V1'] = '1'

import torch
import vllm

# torch.cuda.set_per_process_memory_fraction(0.59, device=None)

MLEN = 30000
# MLEN = 50000

samp = vllm.SamplingParams(
                            # logprobs=0,
                            # prompt_logprobs=0,
                            max_tokens=1)
llm = vllm.LLM(model="meta-llama/Llama-3.1-8B-Instruct",
               enforce_eager=True,
               max_model_len=MLEN + 100,
               gpu_memory_utilization=0.8,
               block_size=16,
               enable_prefix_caching=True,
               max_num_batched_tokens=MLEN+100,
               enable_chunked_prefill=False,)


def tokenid_execution(token_ids):
    header = str(torch.rand(1).item())
    for idx, tokens in enumerate(token_ids):
        assert isinstance(tokens, list)
        llm.llm_engine.add_request(
            request_id=f"{header}-{idx}",
            prompt={'prompt_token_ids': tokens},
            params=samp
        )

    llm.llm_engine.step()

    assert llm.llm_engine.scheduler[0].get_num_unfinished_seq_groups() == 0


prompt_len = 20000
prefix = torch.randint(llm.llm_engine.model_config.get_vocab_size(), size=(prompt_len, )).tolist()

suffixs = [torch.randint(llm.llm_engine.model_config.get_vocab_size(), size=(150, )).tolist() for _ in range(50)]

tokenid_execution([prefix])

import time

profiler = cProfile.Profile()
profiler.enable()


tokenid_execution([prefix + suffix for suffix in suffixs])

profiler.disable()

profiler.dump_stats('generate_profile_isolated.prof')

torch.cuda.synchronize()

