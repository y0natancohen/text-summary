from __future__ import annotations
import time
from pathlib import Path
from typing import List

import ctranslate2
from diskcache import Cache
from transformers import AutoTokenizer

from data_classes import SummaryResultPartial
from md_utils import get_clean_markdown_text_heading
from text_summary_eval_interface import TextSummaryEvalInterface, DATA_SET
cache = Cache("~/PycharmProjects/tavily/.diskcache")


# Model configuration
MODEL_ID = "google/flan-t5-small"
CT2_DIR = Path("ct2_models") / "flan-t5-small-int8f16"
QUANTIZATION = "int8_float16"
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"
FLASH_ATTENTION = False
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Summarization hyperparameters
MAX_INPUT_TOKENS = 512
CHUNK_SIZE = 420
OVERLAP = 60
PER_CHUNK_MAX_NEW_TOKENS = 150
BATCH_SIZE = 100
BEAM_SIZE = 1
REDUCE_GROUP_SIZE = 3

# Prompt templates
SUMMARY_PROMPT = \
"""provide a concise summary of this website, Keep it factual, keep important key-words, names, identities and important information, ignore any links:
[BEGIN DATA]
{text}
[END DATA]"""


def _chunk_ids(ids: List[int], chunk_size: int, overlap: int) -> List[List[int]]:
    """Split token IDs into overlapping chunks."""
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
    """Fast batched generation with CTranslate2."""
    outputs: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        batch_tokens = []
        for p in batch_prompts:
            ids = tokenizer.encode(p, truncation=True, max_length=max_input_tokens)
            toks = tokenizer.convert_ids_to_tokens(ids)
            batch_tokens.append(toks)

        results = translator.translate_batch(
            batch_tokens,
            beam_size=beam_size,
            return_scores=False,
            max_input_length=max_input_tokens,
            max_decoding_length=max_new_tokens,
        )

        for r in results:
            out_tokens = r.hypotheses[0]
            out_ids = tokenizer.convert_tokens_to_ids(out_tokens)
            text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            outputs.append(text)

    return outputs


# @cache.memoize()
def summary_flan_t5(text: str) -> str:
    """enforce 1.5k chars"""
    summary = _summary_flan_t5(text)
    for i in range(5):
        if len(summary) < 1500:
            return summary
        else:
            print(f're running flan for the {i}th time, length={len(summary)}')
            summary = _summary_flan_t5(summary)
    else:
        return summary[:1500]

# @cache.memoize()
def _summary_flan_t5(text: str) -> str:

    try:
        translator = ctranslate2.Translator(
            str(CT2_DIR),
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            flash_attention=FLASH_ATTENTION,
        )
    except TypeError:
        translator = ctranslate2.Translator(str(CT2_DIR), device=DEVICE, compute_type=COMPUTE_TYPE)

    # Chunk-based summarization with hierarchical reduction
    full_ids = tokenizer.encode(text, truncation=False)
    chunks_ids = _chunk_ids(full_ids, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

    # Decode chunks and build prompts
    chunk_texts = [tokenizer.decode(c, skip_special_tokens=True) for c in chunks_ids]
    chunk_prompts = [SUMMARY_PROMPT.format(text=ct) + ct for ct in chunk_texts]
    # for cp in chunk_prompts:
        # print('-- chunk prompt --')
        # print(cp)
    # Summarize all chunks (batched)
    chunk_summaries = _ct2_generate_batch(
        translator,
        tokenizer,
        chunk_prompts,
        max_input_tokens=MAX_INPUT_TOKENS,
        max_new_tokens=PER_CHUNK_MAX_NEW_TOKENS,
        beam_size=BEAM_SIZE,
        batch_size=BATCH_SIZE,
    )
    # for cs in chunk_summaries:
        # print('-- chunk summary --')
        # print(cs)

    return '\n'.join(chunk_summaries)


class FlanT5Summarizer(TextSummaryEvalInterface):
    def _summarize(self, text: str) -> SummaryResultPartial:
        # print('flan t5 summary start')
        # clean_text = get_clean_markdown_text_heading(text)
        summary = summary_flan_t5(text)
        # print('flan t5 summary end')
        return SummaryResultPartial(summary=summary, cost=0.)

# if __name__ == "__main__":
#     text = DATA_SET[44]['markdown_content']
#     s = summary_flan_t5(text)
#     print(s)
