# Qwen Inference Optimization Analysis & Recommendations

## Current Code Review

### Current Optimizations Already Applied:
1. ✅ **4-bit Quantization** (`load_in_4bit=True`) - Reduces memory and speeds up computation
2. ✅ **torch.compile** - Compiles model for faster execution
3. ✅ **Greedy Decoding** (`do_sample=False`, `num_beams=1`) - Fastest decoding strategy
4. ✅ **KV Cache** (`use_cache=True`) - Reuses past key-value pairs
5. ✅ **Warmup Function** - Pre-compiles the model before actual inference

### Areas for Improvement:

#### 1. **Attention Implementation**
Currently using default attention. Consider:
- **Flash Attention 2**: Reduces memory traffic and speeds up attention computation
- **SDPA (Scaled Dot Product Attention)**: Already mentioned in comments, should be enabled

#### 2. **torch.compile Configuration**
Current: `torch.compile(model)` (default mode)
- Consider `mode="reduce-overhead"` for inference workloads
- Consider `mode="max-autotune"` if compilation time is acceptable

#### 3. **Generation Parameters**
- `max_new_tokens=180` - Consider if this is optimal for your use case
- Missing `pad_token_id` - Should be set explicitly
- Consider using `cache_implementation="static"` for torch.compile auto-trigger

#### 4. **Model Loading**
- Consider using `torch_dtype=torch.float16` explicitly (even with 4-bit, compute dtype matters)
- Consider `low_cpu_mem_usage=True` for faster loading

---

## Recommended Optimization Techniques

### 1. **Flash Attention 2** (High Priority)
Flash Attention 2 tiles attention computations and avoids large intermediate tensors, reducing memory footprint and increasing speed.

**Implementation:**
```python
# Install: pip install flash-attn
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    attn_implementation="flash_attention_2",  # Add this
)
```

**Alternative (if Flash Attention 2 not available):**
```python
attn_implementation="sdpa"  # Scaled Dot Product Attention (built-in)
```

### 2. **Optimized torch.compile Mode**
```python
model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
```

**Modes:**
- `"default"` - Balanced (current)
- `"reduce-overhead"` - Reduces Python overhead, faster inference
- `"max-autotune"` - Fastest but longer compilation time

### 3. **Static KV Cache for torch.compile**
Using static cache triggers torch.compile automatically in generate():
```python
output_ids = model.generate(
    **inputs,
    max_new_tokens=180,
    do_sample=False,
    num_beams=1,
    use_cache=True,
    cache_implementation="static",  # Add this
    pad_token_id=tokenizer.pad_token_id,  # Add this
)
```

### 4. **Batch Processing** (if processing multiple texts)
If you're processing multiple texts, batch them together:
```python
# Process multiple texts at once
texts = [text1, text2, text3]
prompts = [create_prompt(t) for t in texts]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
outputs = model.generate(**inputs, ...)
```

### 5. **Speculative Decoding** (Advanced)
Use a smaller draft model to generate candidate tokens, then verify with the main model. Can provide 2-3x speedup.

### 6. **Continuous Batching** (For Serving)
If building a serving application, use `generate_batch()` for dynamic batching.

---

## Chunking Strategy for Long Text Summarization

### Problem
When `markdown_content` is very long (exceeds model's context window or causes slow inference), chunking is necessary.

### Strategy: Hierarchical Chunking with Overlap

#### Approach 1: Sliding Window with Overlap (Recommended)
Split long text into overlapping chunks, summarize each, then combine summaries.

```python
def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
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
    
    return chunks


def summarize_chunked(text: str, model, tokenizer, chunk_size: int = 2000, overlap: int = 200) -> str:
    """
    Summarize long text by chunking.
    
    Strategy:
    1. Split text into overlapping chunks
    2. Summarize each chunk independently
    3. Combine chunk summaries into final summary
    """
    # Check if chunking is needed
    if len(text) <= chunk_size:
        return summarize(text, model, tokenizer)
    
    # Split into chunks
    chunks = chunk_text(text, chunk_size, overlap)
    
    # Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        summary = summarize(chunk, model, tokenizer)
        chunk_summaries.append(summary)
    
    # Combine summaries
    combined_text = ' '.join(chunk_summaries)
    
    # If combined summary is still long, summarize again
    if len(combined_text) > chunk_size:
        return summarize(combined_text, model, tokenizer)
    
    return combined_text
```

#### Approach 2: Hierarchical Summarization (For Very Long Texts)
For extremely long texts, use a two-level hierarchy:

```python
def hierarchical_summarize(text: str, model, tokenizer, 
                          first_level_chunk_size: int = 4000,
                          second_level_chunk_size: int = 2000) -> str:
    """
    Two-level hierarchical summarization:
    1. First level: Summarize large chunks
    2. Second level: Summarize the first-level summaries
    """
    # Level 1: Split into large chunks and summarize
    level1_chunks = chunk_text(text, first_level_chunk_size, overlap=400)
    level1_summaries = [summarize(chunk, model, tokenizer) for chunk in level1_chunks]
    
    # Level 2: Combine and summarize again
    combined = ' '.join(level1_summaries)
    
    if len(combined) > second_level_chunk_size:
        level2_chunks = chunk_text(combined, second_level_chunk_size, overlap=200)
        level2_summaries = [summarize(chunk, model, tokenizer) for chunk in level2_chunks]
        return ' '.join(level2_summaries)
    
    return summarize(combined, model, tokenizer)
```

#### Approach 3: Semantic Chunking (Advanced)
Split based on semantic boundaries (paragraphs, sections) rather than fixed sizes:

```python
def semantic_chunk_text(text: str, max_chunk_size: int = 2000) -> list[str]:
    """
    Split text at semantic boundaries (paragraphs, headings).
    """
    chunks = []
    
    # Split by double newlines (paragraphs)
    paragraphs = text.split('\n\n')
    
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para) + 2  # +2 for '\n\n'
        
        if current_length + para_length > max_chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks
```

### Chunking Parameters Recommendations

1. **Chunk Size**: 
   - For Qwen2.5-1.5B: ~1500-2000 tokens (consider model's max context)
   - Leave room for prompt + summary tokens
   - Formula: `chunk_size = max_context - prompt_tokens - max_summary_tokens - safety_margin`

2. **Overlap Size**:
   - 10-20% of chunk size (e.g., 200 tokens for 2000 token chunks)
   - Ensures context continuity between chunks

3. **When to Chunk**:
   - If input text > 80% of model's context window
   - If inference time > acceptable threshold
   - If memory usage is high

### Chunking Implementation Notes

1. **Preserve Context**: Use overlap to maintain context between chunks
2. **Sentence Boundaries**: Split at sentence boundaries when possible
3. **Markdown Structure**: For markdown, consider splitting at heading boundaries (`#`, `##`)
4. **Parallel Processing**: Can process chunks in parallel if using batch processing
5. **Final Combination**: May need to summarize combined summaries if still too long

---

## Implementation Priority

### High Priority (Easy Wins):
1. ✅ Enable SDPA or Flash Attention 2
2. ✅ Use `mode="reduce-overhead"` for torch.compile
3. ✅ Add `cache_implementation="static"`
4. ✅ Set `pad_token_id` explicitly

### Medium Priority:
5. Implement chunking strategy for long texts
6. Add batch processing if applicable
7. Optimize prompt length

### Low Priority (Advanced):
8. Speculative decoding
9. Continuous batching (if serving)
10. Model distillation (if accuracy allows)

---

## Expected Performance Improvements

- **Flash Attention 2**: 1.5-2x speedup, 30-50% memory reduction
- **torch.compile reduce-overhead**: 10-20% additional speedup
- **Static KV Cache**: 5-10% speedup, enables torch.compile in generate()
- **Chunking**: Enables processing of texts longer than context window
- **Combined optimizations**: 2-3x overall speedup possible

---

## Testing Recommendations

1. Benchmark each optimization individually
2. Measure: latency, memory usage, summary quality (ROUGE scores)
3. Test with various text lengths (short, medium, long)
4. Compare with baseline (current implementation)

