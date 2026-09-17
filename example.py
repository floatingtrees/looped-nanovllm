import os
from enum import Enum
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


class Model(Enum):
    OURO = "~/huggingface/Ouro-1.4B/"
    QWEN3_8B = "~/huggingface/Qwen3-8B/"
    QWEN3_0_6B = "~/huggingface/Qwen3-0.6B/"

    @property
    def path(self) -> str:
        return os.path.expanduser(self.value)


def main():
    path = Model.QWEN3_8B.path
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ] * 64
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
        break


if __name__ == "__main__":
    main()
