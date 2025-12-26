from __future__ import annotations
import argparse
import time
from pathlib import Path

import ctranslate2
from ctranslate2.converters import TransformersConverter
from transformers import AutoTokenizer

from md_utils import get_clean_markdown_text_heading
from data_set import DATA_SET

MODEL_ID = "google/flan-t5-small"


def maybe_convert_to_ct2(model_id: str, out_dir: Path, quantization: str, force: bool = False) -> None:
    """
    Converts a Hugging Face Transformers model to CTranslate2 format (one-time step).
    """
    if out_dir.exists() and ctranslate2.contains_model(str(out_dir)):
        return

    out_dir.parent.mkdir(parents=True, exist_ok=True)

    converter = TransformersConverter(model_id)
    # quantization can be: int8, int8_float16, float16, float32, etc.
    # See CTranslate2 docs for supported values.
    converter.convert(str(out_dir), quantization=quantization, force=force)


def pick_device(requested: str) -> str:
    if requested:
        return requested
    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


def pick_compute_type(device: str, requested: str) -> str:
    """
    Tries to choose a compute type supported on this system/device.
    """
    if requested in (None, "", "auto"):
        return "auto"

    supported = set(ctranslate2.get_supported_compute_types(device))
    if requested in supported:
        return requested

    # sensible fallbacks
    for fallback in ["int8_float16", "float16", "int8", "float32"]:
        if fallback in supported:
            return fallback

    return "default"


# def summarize(
#     translator: ctranslate2.Translator,
#     tokenizer: AutoTokenizer,
#     text: str,
#     max_input_tokens: int,
#     max_new_tokens: int,
#     beam_size: int,
# ) -> str:
#     # FLAN-T5 follows the T5 "task prefix" convention; for summarization use "summarize: ..."
#     prompt = f"summarize: {text}"
#
#     # IMPORTANT: flan-t5-small has a short context window; truncation keeps it safe.
#     input_ids = tokenizer.encode(prompt, truncation=True, max_length=max_input_tokens)
#     input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
#
#     results = translator.translate_batch(
#         [input_tokens],
#         beam_size=beam_size,                 # 1 = greedy (fastest)
#         return_scores=False,                 # keep False for max speed
#         max_input_length=max_input_tokens,   # truncate at runtime too
#         max_decoding_length=max_new_tokens,  # cap output length for latency
#     )
#
#     output_tokens = results[0].hypotheses[0]
#     output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
#     return tokenizer.decode(output_ids, skip_special_tokens=True)


from typing import List, Optional
import math

import ctranslate2
from transformers import AutoTokenizer


def _chunk_ids(ids: List[int], chunk_size: int, overlap: int) -> List[List[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks = []
    start = 0
    n = len(ids)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(ids[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def _ct2_generate_batch(
    translator: ctranslate2.Translator,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_input_tokens: int,
    max_new_tokens: int,
    beam_size: int = 1,
    batch_size: int = 8,
) -> List[str]:
    """
    Fast batched generation with CTranslate2.
    - prompts are strings; we tokenize to CT2 tokens
    - greedy decoding (beam_size=1) is fastest
    """
    outputs: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        # Tokenize each prompt to ids (truncate to max_input_tokens for safety)
        batch_tokens = []
        for p in batch_prompts:
            print('--- prompt to summarize mini text: ---')
            print(p)
            ids = tokenizer.encode(p, truncation=True, max_length=max_input_tokens)
            toks = tokenizer.convert_ids_to_tokens(ids)
            batch_tokens.append(toks)

        results = translator.translate_batch(
            batch_tokens,
            beam_size=beam_size,               # 1 => greedy (fastest)
            return_scores=False,
            max_input_length=max_input_tokens,
            max_decoding_length=max_new_tokens,
        )

        for r in results:
            out_tokens = r.hypotheses[0]
            out_ids = tokenizer.convert_tokens_to_ids(out_tokens)
            text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            outputs.append(text)
            print('---- mini summary -----')
            print(text)
    return outputs


def summarize(
    translator: ctranslate2.Translator,
    tokenizer: AutoTokenizer,
    text: str,
    *,
    chunking: bool = True,
    max_input_tokens: int = 512,          # model input limit
    chunk_size: int = 420,                # keep <512 to leave room for prefix/safety
    overlap: int = 60,
    per_chunk_max_new_tokens: int = 80,   # keep short for speed
    final_max_new_tokens: int = 220,
    batch_size: int = 8,                  # tune based on VRAM; 8 is usually safe for flan-t5-small
    beam_size: int = 1,                   # keep 1 for fastest
) -> str:
    """
    Summarize a document. If chunking=True, uses map-reduce with hierarchical reduction.

    Speed tips:
      - beam_size=1 (greedy)
      - per_chunk_max_new_tokens small (60–120)
      - batch_size as large as VRAM allows
    """
    text = text.strip()
    if not text:
        return ""

    # Single-shot (only safe for short texts)
    summary_prompt = """Please provide a concise summary of the following text, Keep it factual and keep important keywords:\n\n{text}"""
    if not chunking:
        prompt = summary_prompt.format(text=text)
        return _ct2_generate_batch(
            translator, tokenizer, [prompt],
            max_input_tokens=max_input_tokens,
            max_new_tokens=final_max_new_tokens,
            beam_size=beam_size,
            batch_size=1,
        )[0]

    # --- 1) Chunk the full document in *token space* (no truncation here) ---
    full_ids = tokenizer.encode(text, truncation=False)
    chunks_ids = _chunk_ids(full_ids, chunk_size=chunk_size, overlap=overlap)

    # Decode chunk ids back to text and build prompts
    chunk_texts = [tokenizer.decode(c, skip_special_tokens=True) for c in chunks_ids]
    chunk_prompts = [summary_prompt.format(text=ct) + ct for ct in chunk_texts]

    # --- 2) Map: summarize all chunks (batched) ---
    chunk_summaries = _ct2_generate_batch(
        translator,
        tokenizer,
        chunk_prompts,
        max_input_tokens=max_input_tokens,
        max_new_tokens=per_chunk_max_new_tokens,
        beam_size=beam_size,
        batch_size=batch_size,
    )

    # --- 3) Reduce: hierarchical summarize summaries until it fits ---
    reduce_summary_prompt = """"please combine and deduplicate these points into a concise summary, Keep it factual:\n{text}"""
    def summaries_fit(summaries: List[str]) -> bool:
        combined = "\n".join(f"- {s}" for s in summaries)
        ids = tokenizer.encode(reduce_summary_prompt.format(text=combined), truncation=False)
        return len(ids) <= max_input_tokens

    current = chunk_summaries

    # If many chunk summaries, reduce in groups
    # group_size is chosen so that concatenated bullet list likely fits under 512 tokens
    # group_size = max(4, min(12, math.floor(max_input_tokens / 60)))  # heuristic
    # group_size = 3
    #
    # while not summaries_fit(current):
    #     grouped: List[List[str]] = []
    #     for i in range(0, len(current), group_size):
    #         grouped.append(current[i : i + group_size])
    #
    #     reduce_prompts = []
    #     for g in grouped:
    #         blob = "\n".join(f"- {s}" for s in g)
    #         reduce_prompts.append(reduce_summary_prompt.format(text=blob))
    #         print(f'----- reduce prompt -----')
    #         print(blob)
    #
    #     current = _ct2_generate_batch(
    #         translator,
    #         tokenizer,
    #         reduce_prompts,
    #         max_input_tokens=max_input_tokens,
    #         max_new_tokens=final_max_new_tokens,  # keep reductions short too
    #         beam_size=beam_size,
    #         batch_size=batch_size,
    #     )
    #
    #     # --- 4) Final pass ---
    #     print('---------- small chunk summaries to combine:')
    #     for s in current:
    #         print(s)
    #
    # final_blob = "\n".join(f"- {s}" for s in current)
    # final_prompt = (
    #         "provide a concise summary of the following text, Keep it factual and remove redundancy:\n"
    #         + final_blob
    # )
    #
    # final = _ct2_generate_batch(
    #     translator,
    #     tokenizer,
    #     [final_prompt],
    #     max_input_tokens=max_input_tokens,
    #     max_new_tokens=final_max_new_tokens,
    #     beam_size=beam_size,
    #     batch_size=1,
    # )[0]

    # return final
    return '- \n'.join(chunk_summaries)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--text", type=str, default=None, help="Text to summarize.")
    # parser.add_argument("--text_file", type=str, default=None, help="Path to a text file to summarize.")
    # parser.add_argument("--ct2_dir", type=str, default=str(Path("ct2_models") / "flan-t5-small-int8f16"))
    # parser.add_argument("--quantization", type=str, default="int8_float16",
    #                     help="CT2 conversion quantization: int8_float16, float16, float32, ...")
    # parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto-detect).")
    # parser.add_argument("--compute_type", type=str, default="int8_float16",
    #                     help="CT2 runtime compute type: int8_float16, float16, auto, ...")
    # parser.add_argument("--flash_attention", action="store_true",
    #                     help="Enable FlashAttention 2 (if supported by your build/GPU).")
    # parser.add_argument("--max_input_tokens", type=int, default=512)
    # parser.add_argument("--max_new_tokens", type=int, default=128)
    # parser.add_argument("--beam_size", type=int, default=1)
    # parser.add_argument("--warmup", action="store_true", help="Run a short warmup call before timing.")
    # args = parser.parse_args()

    # if not args.text and not args.text_file:
    #     raise SystemExit("Provide --text or --text_file")

    import time
    i = 10
    text_md = DATA_SET[i]['markdown_content']
    text, headings = get_clean_markdown_text_heading(text_md)

    # if args.text_file:
    #     text = Path(args.text_file).read_text(encoding="utf-8", errors="ignore")

    ct2_dir = Path("ct2_models") / "flan-t5-small-int8f16"

    # 1) One-time conversion to CT2 format (downloads weights via HF under the hood)
    maybe_convert_to_ct2(MODEL_ID, ct2_dir, quantization='int8_float16', force=False)

    # 2) Tokenizer from Hugging Face (cached locally after first download)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 3) Load CT2 translator
    device = pick_device('cuda')
    compute_type = pick_compute_type(device, 'int8_float16')

    # Try flash attention if requested; fall back gracefully if not supported.
    try:
        translator = ctranslate2.Translator(
            str(ct2_dir),
            device=device,
            compute_type=compute_type,
            # flash_attention=True,
            flash_attention=False,
        )
    except TypeError:
        # Older builds might not support the flash_attention flag
        translator = ctranslate2.Translator(str(ct2_dir), device=device, compute_type=compute_type)

    # 4) Warmup (helps stabilize latency in real-time systems)
    max_input_tokens = 512
    _ = summarize(translator, tokenizer, "Warmup text.",
                  max_input_tokens=max_input_tokens,
                  # max_new_tokens=128,
                  beam_size=1,
                  chunking=False,
                  chunk_size=420,
                  overlap=60,
                  per_chunk_max_new_tokens=80,
                  final_max_new_tokens=220,
                  batch_size=8,
                  )

    # 5) Timed run
    t0 = time.perf_counter()
    summary = summarize(
        translator,
        tokenizer,
        text,
        max_input_tokens=max_input_tokens,
        # max_new_tokens=128,
        beam_size=1,
        chunking=True,
        chunk_size=420,
        overlap=60,
        per_chunk_max_new_tokens=200,
        final_max_new_tokens=300,
        batch_size=10,
    )
    t1 = time.perf_counter()

    print("\n=== SUMMARY ===\n")
    print(summary)
    print(f"\n[Timing] {(t1 - t0)*1000:.1f} ms  | device={device} compute_type={compute_type} "
          f"beam_size={1}")


if __name__ == "__main__":
    main()
