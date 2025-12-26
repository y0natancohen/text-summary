import sys
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from data_set import DATA_SET

# import vllm
# from vllm import LLM, SamplingParams
# Official 4-bit AWQ checkpoint (INT4)
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"

# Tokenizer is used to format the prompt using the model's chat template.
# (Qwen model card shows apply_chat_template usage.)  :contentReference[oaicite:3]{index=3}
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# IMPORTANT speed knobs:
# - quantization="awq" uses the AWQ INT4 checkpoint.
# - dtype="float16" is the safe fast dtype on RTX 3070 (AWQ commonly expects fp16).
# - max_model_len: keep this near what you actually need to reduce KV-cache reservation.
llm = LLM(
    model=MODEL_ID,
    quantization="awq",
    dtype="float16",
    max_model_len=8192,            # raise if you truly need longer prompts
    gpu_memory_utilization=0.4,   # use most of VRAM (tweak down if you get OOM)
)

def build_prompt(text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Summarize accurately and concisely."},
        {"role": "user", "content": "Summarize the following text in 6 bullet points.\n\n" + text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def summarize(text: str, max_tokens: int = 200) -> str:
    prompt = build_prompt(text)

    # Fast decoding: temperature=0 => greedy in vLLM. :contentReference[oaicite:4]{index=4}
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
    )

    outputs = llm.generate([prompt], params)
    return outputs[0].outputs[0].text.strip()

if __name__ == "__main__":
    # Read input from stdin for easy piping:
    # cat mydoc.txt | python qwen_fast_awq_vllm.py
    i = 5
    import time
    markdown_content = DATA_SET[i]['markdown_content']
    start_time = time.time()
    print(summarize(markdown_content, max_tokens=220))
    end_time = time.time()
