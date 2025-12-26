from data_classes import SummaryResultPartial
from text_summary_eval_interface import TextSummaryEvalInterface
from md_utils import get_clean_markdown_text_heading

import re
from typing import List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# -----------------------------
# 1) Multilingual-ish sentence splitter (rule-based heuristics)
# -----------------------------
_SENT_END_CHARS = r"\.\!\?\。\！\？\…\¡\¿\؟\۔"
_BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+", re.UNICODE)

# Abbreviation guard (English-biased, optional)
_ABBR = {
    "e.g.", "i.e.", "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.",
    "etc.", "fig.", "eq.", "dept.", "inc.", "ltd.", "no."
}

# NOTE: not an f-string to avoid the "single } in f-string" problem
_SENT_SPLIT_PATTERN = r'(?<=[%s])(?:["\'\)\]\}»\u201d\u2019]+)?(?:\s+|$)' % _SENT_END_CHARS
_SENT_SPLIT_RE = re.compile(_SENT_SPLIT_PATTERN, flags=re.UNICODE)


def split_sentences_multilingual(text: str, max_line_len: int = 300) -> List[str]:
    """
    Sentence splitting heuristics:
    - Normalize whitespace
    - Treat bullet lines as boundaries
    - Split on multilingual sentence-ending punctuation
    - Merge splits caused by common abbreviations (English-only heuristic)
    - Force-cut very long chunks without punctuation (prevents giant "sentences")
    """
    if not text or not text.strip():
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text).strip()

    # Split into non-empty lines, treat bullet starts as chunk boundaries
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks: List[str] = []
    cur: List[str] = []

    for ln in lines:
        if _BULLET_RE.match(ln) and cur:
            chunks.append(" ".join(cur).strip())
            cur = [ln]
        else:
            cur.append(ln)

        if sum(len(x) for x in cur) > max_line_len:
            chunks.append(" ".join(cur).strip())
            cur = []

    if cur:
        chunks.append(" ".join(cur).strip())

    # Split each chunk into sentences by punctuation boundary
    sentences: List[str] = []
    for c in chunks:
        parts = _SENT_SPLIT_RE.split(c)
        for p in parts:
            p = re.sub(r"\s+", " ", p).strip()
            if p:
                sentences.append(p)

    # Merge if we split after an abbreviation like "e.g."
    merged: List[str] = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        low = s.lower().strip()
        if any(low.endswith(ab) for ab in _ABBR) and i + 1 < len(sentences):
            s = (s + " " + sentences[i + 1]).strip()
            i += 2
        else:
            i += 1
        merged.append(re.sub(r"\s+", " ", s).strip())

    return [s for s in merged if s]


# -----------------------------
# 2) TF-IDF scoring on character n-grams
# -----------------------------
def compute_tfidf_char_ngrams(
    sentences: List[str],
    ngram_range: Tuple[int, int] = (3, 5),
    min_df: int = 1,
):
    """
    Returns:
      X: sparse tf-idf matrix (n_sentences x n_features)
      doc_vec: 1 x n_features vector representing the "document query" (sparse, normalized)
      vectorizer
    """
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=ngram_range, min_df=min_df)
    X = vectorizer.fit_transform(sentences)

    # Robust doc vector: sum then L2-normalize (avoids np.matrix issues)
    doc_sum = X.sum(axis=0)
    # Convert to array if it's a matrix (for older sklearn versions)
    if hasattr(doc_sum, 'A'):
        doc_sum = np.asarray(doc_sum.A)
    else:
        doc_sum = np.asarray(doc_sum)
    doc_vec = normalize(doc_sum)  # 1 x D
    return X, doc_vec, vectorizer


# -----------------------------
# 3) Optional near-duplicate removal
# -----------------------------
def dedupe_sentences(
    sentences: List[str],
    sim_threshold: float = 0.98,
    ngram_range: Tuple[int, int] = (3, 5),
) -> List[int]:
    """
    Returns indices of sentences to keep, removing near-duplicates.
    Greedy: keep sentence i if it's not too similar to any previously kept sentence.
    """
    if len(sentences) <= 1:
        return list(range(len(sentences)))

    X, _, _ = compute_tfidf_char_ngrams(sentences, ngram_range=ngram_range)
    keep: List[int] = []
    for i in range(len(sentences)):
        if not keep:
            keep.append(i)
            continue
        sims = cosine_similarity(X[i], X[keep]).ravel()
        if sims.max(initial=0.0) < sim_threshold:
            keep.append(i)
    return keep


# -----------------------------
# 4) MMR selection (avoid redundancy)
# -----------------------------
def mmr_select_indices(
    sentences: List[str],
    X,
    doc_vec,
    max_chars: int = 1500,
    lambda_param: float = 0.7,
    min_sent_chars: int = 5,
    max_sentences: Optional[int] = None,
) -> List[int]:
    """
    MMR:
      score(i) = λ * sim(i, doc) - (1-λ) * max_{j in selected} sim(i, j)
    Stops when adding next sentence would exceed max_chars.
    """
    valid = [i for i, s in enumerate(sentences) if len(s.strip()) >= min_sent_chars]
    if not valid:
        return []

    Xv = X[valid]

    # Relevance to document
    rel = cosine_similarity(Xv, doc_vec).ravel()

    # Start with most relevant sentence
    first_local = int(np.argmax(rel))
    selected_local = [first_local]
    selected_global = [valid[first_local]]

    remaining = [i for i in range(len(valid)) if i != first_local]

    def join_len(idxs_global: List[int]) -> int:
        return len(" ".join(sentences[i] for i in idxs_global)) if idxs_global else 0

    while remaining:
        if max_sentences is not None and len(selected_global) >= max_sentences:
            break

        # Compute redundancy vs selected only (avoids O(N^2) full precompute)
        X_sel = Xv[selected_local]
        X_rem = Xv[remaining]
        sim_matrix = cosine_similarity(X_rem, X_sel)
        # Handle both matrix and array cases
        if hasattr(sim_matrix, 'A'):
            redundancy = sim_matrix.max(axis=1).A.ravel()
        else:
            redundancy = sim_matrix.max(axis=1).ravel()

        mmr = lambda_param * rel[remaining] - (1.0 - lambda_param) * redundancy
        best_pos = int(np.argmax(mmr))
        best_local = remaining[best_pos]
        best_global = valid[best_local]

        if join_len(selected_global + [best_global]) > max_chars:
            break

        selected_local.append(best_local)
        selected_global.append(best_global)
        remaining.pop(best_pos)

    return selected_global


# -----------------------------
# 5) Stitch + trim
# -----------------------------
def stitch_and_trim(
    sentences: List[str],
    idxs: List[int],
    max_chars: int = 1500,
    add_ellipsis_on_hard_trim: bool = True,
) -> str:
    """
    Join selected sentences in original order, stop before overflow.
    If nothing fits (single huge sentence), hard-trim.
    """
    if not sentences or not idxs:
        return ""

    idxs_sorted = sorted(idxs)
    out_parts: List[str] = []
    cur_len = 0

    for i in idxs_sorted:
        s = sentences[i].strip()
        if not s:
            continue
        add = s if not out_parts else " " + s
        if cur_len + len(add) <= max_chars:
            out_parts.append(s)
            cur_len += len(add)
        else:
            break

    if out_parts:
        return " ".join(out_parts)

    # Hard-trim fallback
    s0 = sentences[idxs_sorted[0]].strip()
    if len(s0) <= max_chars:
        return s0

    trimmed = s0[:max_chars].rstrip()
    if add_ellipsis_on_hard_trim and trimmed:
        trimmed = trimmed[:-1].rstrip() + "…"
    return trimmed


# -----------------------------
# Main entry point
# -----------------------------
def extractive_mmr_summary(
    input_text: str,
    max_chars: int = 1500,
    ngram_range: Tuple[int, int] = (3, 5),
    lambda_param: float = 0.7,
    min_sent_chars: int = 5,
    dedupe: bool = True,
    dedupe_threshold: float = 0.98,
    max_sentences: Optional[int] = None,
    force: bool = False,
) -> str:
    """
    Pipeline:
      - split sentences
      - optional dedupe
      - tf-idf char n-grams
      - mmr selection under char budget
      - stitch + trim

    If force=False and the normalized text is already <= max_chars, returns normalized text unchanged.
    """
    sentences = split_sentences_multilingual(input_text)
    if not sentences:
        return ""

    normalized = " ".join(sentences)
    if (not force) and (len(normalized) <= max_chars):
        return normalized

    if dedupe and len(sentences) > 3:
        keep = dedupe_sentences(sentences, sim_threshold=dedupe_threshold, ngram_range=ngram_range)
        sentences = [sentences[i] for i in keep]

    X, doc_vec, _ = compute_tfidf_char_ngrams(sentences, ngram_range=ngram_range)

    selected = mmr_select_indices(
        sentences,
        X,
        doc_vec,
        max_chars=max_chars,
        lambda_param=lambda_param,
        min_sent_chars=min_sent_chars,
        max_sentences=max_sentences,
    )

    return stitch_and_trim(sentences, selected, max_chars=max_chars)


class MarkdownRulesSummarizer(TextSummaryEvalInterface):
    def _summarize(self, text: str) -> SummaryResultPartial:
        text = get_clean_markdown_text_heading(text)
        return SummaryResultPartial(summary=text, cost=0.)


# if __name__ == "__main__":
#     input_text = """
#     we are testing multilingual sentence splitting.
#     we want to do a summary of this multilingual text.
#     the goal is to create a summary.
#     the goal is to create a summary.
#     This is an English paragraph. It has multiple sentences!
#     هنا نص عربي؟ نعم، لديه جمل متعددة.
#     这是中文。它也能被切分。还有更多句子！
#     """
#     print(extractive_mmr_summary(input_text, max_chars=200))
