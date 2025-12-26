import os
from typing import List, Literal, Optional, Tuple
from diskcache import Cache
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.shared_params import Reasoning
# from openai.types import Reasoning
from pydantic import BaseModel, Field
from tenacity import retry, wait_random_exponential, stop_after_attempt
# from flan_t5_summary import summary_flan_t5

from data_set import DATA_SET

USD_PER_1M_TOKENS = 2.5
cache = Cache("~/PycharmProjects/tavily/.diskcache")

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

class UnsupportedClaim(BaseModel):
    summary_quote: str = Field(..., description="Exact quote from the candidate summary that is unsupported.")
    why_unsupported: str = Field(..., description="Explain briefly why it is unsupported vs the source.")

class KeyPointCheck(BaseModel):
    key_point: str
    coverage: Literal["present", "partial", "missing"]

class SummaryJudgeResult(BaseModel):
    faithfulness: int = Field(..., ge=1, le=10)
    coverage: int = Field(..., ge=1, le=10)
    coherence: int = Field(..., ge=1, le=10)
    conciseness: int = Field(..., ge=1, le=10)
    overall: int = Field(..., ge=1, le=10)

    key_points: List[KeyPointCheck]
    unsupported_claims: List[UnsupportedClaim]
    improvement_suggestions: List[str]
    non_relevant_parts: List[str]

    def to_dict(self):
        return self.model_dump()

JUDGE_INSTRUCTIONS = """You are a meticulous evaluator for a text summarization task.
You must judge the summary ONLY against the provided SOURCE TEXT.
Do NOT use outside knowledge. Do NOT assume facts that are not explicitly supported by the source.

Score dimensions (1–10):
- faithfulness (no hallucinations / incorrect claims)
- coverage (captures the most important points)
- coherence (readable)
- conciseness (no redundant information, appropriate length for a summary)


Critical rules:
- If ANY unsupported factual claim exists, faithfulness <= 4.
- Multiple hallucinations or reverses key facts, overall <= 4.
- Do NOT reward verbosity.

Process internally:
- coverage: go over the SOURCE_TEXT and Extract 5–20 important key points, Mark each as present/partial/missing in the summary.
- faithfulness: go over each claim in the SUMMARY TO EVALUATE, List every unsupported claim.
- conciseness: go over each claim in the SUMMARY TO EVALUATE, and decide if its relevant for the summary.
- conciseness: go over each sentence in the SUMMARY TO EVALUATE, and decide if it can be shorter without losing relevant information.
- coherence: go over SUMMARY TO EVALUATE, and assess overall readability and clarity.

Return ONLY the JSON object matching the schema.
"""


USER_PROMPT = """SOURCE TEXT:
<<<{source_text}>>>

SUMMARY TO EVALUATE:
<<<{summary}>>>"""

def final_score(summary_result: SummaryJudgeResult) -> float:
    weights = {
        'faithfulness': 0.4,
        'coverage': 0.3,
        'coherence': 0.1,
        'conciseness': 0.1,
        'overall': 0.1,
    }
    score = (
        summary_result.faithfulness * weights['faithfulness'] +
        summary_result.coverage * weights['coverage'] +
        summary_result.coherence * weights['coherence'] +
        summary_result.conciseness * weights['conciseness'] +
        summary_result.overall * weights['overall']
    )
    score_normalised = (score - 1) / 9
    return score_normalised


# reasoning_profile = Reasoning(effort='minimal')
reasoning_profile = Reasoning(effort='high')



# @cache.memoize()
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def judge_summary(source_text: str, summary: str) -> Tuple[float, float, SummaryJudgeResult]:
    user_payload = USER_PROMPT.format(source_text=source_text, summary=summary)

    # Temperature 0 for consistency; consider also setting a seed for (mostly) reproducible outputs.
    # Reproducibility guidance: set seed + keep params identical; track system_fingerprint. :contentReference[oaicite:7]{index=7}
    resp = client.responses.parse(
        model="gpt-5.2",  # example; choose an eval-appropriate model for your needs
        # model="gpt-5-nano",  # example; choose an eval-appropriate model for your needs
        input=[
            {"role": "developer", "content": JUDGE_INSTRUCTIONS},
            {"role": "user", "content": user_payload},
        ],
        # temperature=0,  # not for gpt-5.2
        text_format=SummaryJudgeResult,
        reasoning=reasoning_profile,  # `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`.

    )
    usage = resp.usage  # includes input_tokens, output_tokens, total_tokens
    total_tokens = usage.total_tokens
    cost_usd = total_tokens * (USD_PER_1M_TOKENS / 1_000_000)
    score = final_score(resp.output_parsed)
    return score, cost_usd, resp.output_parsed


if __name__ == '__main__':
    # text = DATA_SET[44]['markdown_content']
    # print('loaded')
    # summary_t5 = summary_flan_t5(text)
    # print('did summary')
    # reference_summary = DATA_SET[44]['summary']
    # result, cost = judge_summary(DATA_SET[44]['markdown_content'], candidate_summary=summary_t5)
    # print(cost)
    # print(result.model_dump_json(indent=2))
    # print('--------------------')
    # result, cost = judge_summary(DATA_SET[44]['markdown_content'], candidate_summary=reference_summary)

    result, cost, all_result = judge_summary('I have 2 things: cat, dog. I have cat and dog',
                                             summary='I have cat and dog')
    print(result)
    print(cost)
    print(all_result.model_dump_json(indent=2))
