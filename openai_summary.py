import os
import time
from typing import Tuple

from diskcache import Cache
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.shared_params import Reasoning
from tenacity import retry, wait_random_exponential, stop_after_attempt

from data_classes import SummaryResultPartial
from data_set import DATA_SET
from openai_llm_as_a_judge import judge_summary

from text_summary_eval_interface import TextSummaryEvalInterface
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
cache = Cache("~/PycharmProjects/tavily/.diskcache")
reasoning_profile = Reasoning(effort='low')  # "none", "minimal", "low", "medium", "high", "xhigh"


USD_PER_1M_TOKENS = 1.1

# @cache.memoize()
def summarize_text_openai_single_call(text: str) -> Tuple[str, float]:
    if len(text) < 380000:
        return _summarize_text_openai_single_call(text)
    else:
        # do chunking for very large texts
        chunk_size = 350000
        total_cost = 0.0
        
        # Split text into chunks
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary, cost = _summarize_text_openai_single_call(chunk)
            chunk_summaries.append(summary)
            total_cost += cost
        
        # Combine chunk summaries and do final pass
        combined_summaries = "\n\n".join(chunk_summaries)
        final_summary, final_cost = _summarize_text_openai_single_call(combined_summaries)
        total_cost += final_cost
        
        return final_summary, total_cost

# @cache.memoize()
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def _summarize_text_openai_single_call(text: str) -> Tuple[str, float]:
    client = OpenAI(api_key=api_key)
    
    resp = client.responses.parse(
        model="gpt-4.1-nano",  # example; choose an eval-appropriate model for your needs
        # model="gpt-5-nano",  # example; choose an eval-appropriate model for your needs
        input=[
            {"role": "user", "content": f"Please provide a concise summary of the following text:\n\n{text}"},
        ],
        # temperature=0,  # not for gpt-5.2
        # text_format=SummaryOutput,
        # reasoning=reasoning_profile,  # `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`.
        max_output_tokens=350
    )
    
    usage = resp.usage  # includes input_tokens, output_tokens, total_tokens
    total_tokens = usage.total_tokens
    cost_usd = total_tokens * (USD_PER_1M_TOKENS / 1_000_000)
    
    return resp.output_text, cost_usd


class OpenaiSingleCallSummarizer(TextSummaryEvalInterface):
    def _summarize(self, text: str) -> SummaryResultPartial:
        summary, cost = summarize_text_openai_single_call(text)
        return SummaryResultPartial(summary=summary, cost=cost)



def main():
    i = 0
    markdown_content = DATA_SET[i]['markdown_content']
    start_time = time.time()
    summary, cost_usd = summarize_text_openai_single_call(markdown_content)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Cost: ${cost_usd:.6f}")
    print("Summary:")
    print(summary)
    score, judge_cost_usd, summary_judge_result = judge_summary(markdown_content, summary)
    print(f'Score: {score}, Judge Cost: ${judge_cost_usd:.6f}')
    print(f'Summary Judge Result: {summary_judge_result.to_dict()}')


if __name__ == "__main__":
    main()
