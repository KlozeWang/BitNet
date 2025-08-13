import os
from bitnet_vllm.llm import LLM
import torch
import numpy as np
from bitnet_vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer

from datasets import load_dataset

def format(item):
    prompt = ""
    if item["instruction"] != "":
        prompt += "### Instruction\n " + item["instruction"] + "\n"
    if item["input"] != "":
        prompt += "### Input\n " + item["input"] + "\n"
    prompt += "### Response\n "
    return prompt

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    data = load_dataset("yahma/alpaca-cleaned", split="train").select(range(1,20))

    local_rank = 0
    device = f"cuda:{local_rank}"

    path = os.path.expanduser("./checkpoints/")
    llm = LLM(path, local_rank=local_rank, device=device, enforce_eager=False, chat_format=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    prompts = [
        format(item) for item in data
    ]
    outputs = llm.generate(prompts, sampling_params, chat_format=True)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output['text']}")

    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == "__main__":
    main()
