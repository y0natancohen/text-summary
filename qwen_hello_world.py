import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_set import DATA_SET

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print('loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
print('tokenizer loaded')

# RTX 3070 laptop: FP16 is the right default (bf16 usually not supported on 30-series)
print('loading model...')
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     attn_implementation="sdpa",  # try "flash_attention_2" if you install flash-attn
# )
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    load_in_4bit=True,              # requires bitsandbytes
    bnb_4bit_compute_dtype=torch.float16,
)
print('loaded')

print('compiling model...')
model = torch.compile(model)
print('compiled')

def summarize(text: str) -> str:
    prompt = (
        "You are a helpful assistant. Summarize the following text clearly.\n\n"
        "Text:\n"
        f"{text}\n\n"
        "Summary:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,   # deterministic + usually fastest
            num_beams=1,       # greedy decoding for low latency
            use_cache=True,  # important
        )

    # Only return newly generated tokens (not the prompt)
    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def warmup(model, tokenizer, device):
    start_time = time.time()

    model.eval()
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=8, do_sample=False, num_beams=1)
    torch.cuda.synchronize()

    end_time = time.time()
    print(f"warmup Time taken: {end_time - start_time:.2f} seconds")



def main():
    i = 5
    print('----- Markdown Content -----')
    print(DATA_SET[i]['markdown_content'])
    print("----- Summary -----")
    print(DATA_SET[i]['summary'])
    print("----- Qwen Summary -----")
    warmup(model, tokenizer, model.device)
    start_time = time.time()
    print(summarize(DATA_SET[i]['markdown_content']))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
