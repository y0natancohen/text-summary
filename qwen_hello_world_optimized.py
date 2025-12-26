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

# OPTIMIZATION 1: Use SDPA (Scaled Dot Product Attention)
# Note: Flash Attention 2 requires CUDA 11.7+ and compilation, which may not be available.
# SDPA is built into PyTorch and provides excellent performance without compilation.
# If you have CUDA 11.7+ and want to try Flash Attention 2, uncomment the try/except block below.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    load_in_4bit=True,              # requires bitsandbytes
    bnb_4bit_compute_dtype=torch.float16,
    attn_implementation="sdpa",  # Scaled Dot Product Attention (built-in, no compilation needed)
)
print('loaded with SDPA (optimized attention)')

# Alternative: Try Flash Attention 2 if available (requires CUDA 11.7+ and flash-attn package)
# Uncomment below if you have flash-attn installed:
# try:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         device_map="auto",
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         attn_implementation="flash_attention_2",
#     )
#     print('loaded with Flash Attention 2')
# except Exception as e:
#     print(f'Flash Attention 2 not available: {e}')
#     print('Falling back to SDPA...')
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         device_map="auto",
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         attn_implementation="sdpa",
#     )
#     print('loaded with SDPA')

# OPTIMIZATION 2: Use reduce-overhead mode for inference
print('compiling model...')
model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
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
            use_cache=True,    # important
            cache_implementation="static",  # OPTIMIZATION 3: Static cache for torch.compile
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,  # OPTIMIZATION 4: Explicit pad token
        )

    # Only return newly generated tokens (not the prompt)
    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks at sentence boundaries.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    
    # Try to split at sentence boundaries for better context
    sentences = text.split('. ')
    
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence) + 2  # +2 for '. '
        
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append('. '.join(current_chunk) + '.')
            
            # Start new chunk with overlap (last N sentences)
            overlap_sentences = current_chunk[-overlap//50:] if len(current_chunk) > overlap//50 else current_chunk
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s) + 2 for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add final chunk
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks if chunks else [text]


def summarize_chunked(text: str, chunk_size: int = 2000, overlap: int = 200) -> str:
    """
    Summarize long text by chunking.
    
    Strategy:
    1. Split text into overlapping chunks
    2. Summarize each chunk independently
    3. Combine chunk summaries into final summary
    
    Args:
        text: Input text to summarize
        chunk_size: Target size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks
    
    Returns:
        Combined summary
    """
    # Check if chunking is needed (rough estimate: 1 char ≈ 0.25 tokens)
    # For Qwen2.5-1.5B, context window is typically 32K tokens
    # Use chunk_size to stay well under this limit
    if len(text) <= chunk_size:
        return summarize(text)
    
    print(f"Text is long ({len(text)} chars), chunking into smaller pieces...")
    
    # Split into chunks
    chunks = chunk_text(text, chunk_size, overlap)
    print(f"Split into {len(chunks)} chunks")
    
    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        summary = summarize(chunk)
        chunk_summaries.append(summary)
    
    # Combine summaries
    combined_text = ' '.join(chunk_summaries)
    
    # If combined summary is still long, summarize again
    if len(combined_text) > chunk_size:
        print("Combined summary is still long, summarizing again...")
        return summarize(combined_text)
    
    return combined_text


def warmup(model, tokenizer, device):
    start_time = time.time()

    model.eval()
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        _ = model.generate(
            **inputs, 
            max_new_tokens=8, 
            do_sample=False, 
            num_beams=1,
            cache_implementation="static",
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()

    end_time = time.time()
    print(f"warmup Time taken: {end_time - start_time:.2f} seconds")


def main():
    i = 5
    print('----- Markdown Content -----')
    markdown_content = DATA_SET[i]['markdown_content']
    print(f"Length: {len(markdown_content)} characters")
    print(markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content)
    print("----- Reference Summary -----")
    print(DATA_SET[i]['summary'])
    print("----- Qwen Summary -----")
    
    warmup(model, tokenizer, model.device)
    
    start_time = time.time()
    # Use chunked summarization for long texts
    summary = summarize_chunked(markdown_content, chunk_size=200000000, overlap=200)
    end_time = time.time()
    
    print(summary)
    print(f"\nTime taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

