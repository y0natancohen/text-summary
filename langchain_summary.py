import os
from typing import Tuple

from diskcache import Cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from data_classes import SummaryResultPartial
from openai_summary import summarize_text_openai_single_call
from openai_llm_as_a_judge import SummaryJudgeResult, JUDGE_INSTRUCTIONS, USER_PROMPT, final_score, judge_summary
from data_set import DATA_SET
from text_summary_eval_interface import TextSummaryEvalInterface

cache = Cache("~/PycharmProjects/tavily/.diskcache")


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", JUDGE_INSTRUCTIONS),
        ("human", USER_PROMPT),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "make **only necessary changes** to the summary: "
            "only if there is some explicit request to change, add or edit things.\n"
            "Rules:\n"
            "- MUST be faithful to the source. Do not invent details.\n"
            "- Fix issues described in the evaluator feedback.\n"
            "- Add missing key points (if important) but stay concise.\n"
            "- Remove/avoid unsupported claims.\n"
            "- Output only the improved summary text.\n"
            "- total output length should be less then 1500 chars (about 350 tokens).\n",
        ),
        (
            "human",
            "SOURCE TEXT:\n{source}\n\nCURRENT SUMMARY:\n{summary}\n\n"
            "EVALUATOR FEEDBACK:\n"
            "- faithfulness score: {faithfulness}, if faithfulness score is 7 or smaller, fix the unsupported claims in the CURRENT SUMMARY\n"
            "- coverage score: {coverage} if coverage score is 7 or smaller, need to go over the SOURCE TEXT and add missing key points to the CURRENT SUMMARY\n"
            "- coherence: {coherence} if coherence score is 7 or smaller, need to go over the CURRENT SUMMARY and improve readability only without content changing\n"
            "- conciseness: {conciseness} if conciseness score is 7 or smaller, need to go over the CURRENT SUMMARY and"
            " remove non-relevant parts, or - if possible - making sentences shorter and more concise without losing relevant information \n"
            "- Missing key points to consider: {missing_key_points}\n"
            "- Valid key points to keep: {good_key_points} you MUST include these points in the improved summary\n"
            "- Unsupported claims to remove/fix: {unsupported_claims}\n"
            "- general improvement suggestions: {improvement_suggestions}\n"
            "- non relevant parts: {improvement_suggestions}\n"
            "- CURRENT SUMMARY: total length: {summary_length}, max allowed length is 1500 chars (350 tokens)"
            " if summary_length is larger then 1500 chars (about 350 tokens) then"
            " YOU MUST make the summary shorter, and decide on whats most important to keep.\n\n"
            "Write an improved summary:",
        ),
    ]
)
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
MODEL = 'gpt-4o'

def iterative_summary_agent(source_text: str, initial_summary: str, max_iters: int, threshold: float,
                            model: str) -> Tuple[str, list, float]:

    writer = ChatOpenAI(model=model, temperature=0.1, api_key=api_key)
    # Deterministic judging is helpful for stable loops
    # judge_llm_result = ChatOpenAI(model=model, temperature=0, api_key=api_key).with_structured_output(SummaryJudgeResult)
    # judge_chain = JUDGE_PROMPT | judge_llm_result
    rewrite_chain = REWRITE_PROMPT | writer

    summary = initial_summary

    history = []
    total_rewrite_cost = 0.0

    for i in range(1, max_iters + 1):
        print(f'agentic summary loop {i} start')
        overall_score, cost_usd, eval_result = judge_summary(source_text, summary)
        # eval_result: SummaryJudgeResult = judge_chain.invoke({"source_text": source_text, "summary": summary})

        # unsupported_list = [f'- {uc.summary_quote} -> {uc.why_unsupported}' for uc in eval_result.unsupported_claims]
        unsupported_list = [f'- {uc.summary_quote} -> {uc.why_unsupported}' for uc in eval_result.unsupported_claims]
        unsupported_list_str = "\n".join(unsupported_list) if unsupported_list else "None"
        good_key_points = [f'- {kp.key_point}' for kp in eval_result.key_points if kp.coverage == 'present']
        good_key_points_str = "\n".join(good_key_points) if good_key_points else "None"
        missing_key_points = [f'- {kp.key_point}' for kp in eval_result.key_points if kp.coverage in ["partial", "missing"]]
        missing_key_points_str = "\n".join(missing_key_points) if missing_key_points else "None"
        improvement_suggestions_str = "\n".join(eval_result.improvement_suggestions) if eval_result.improvement_suggestions else "None"
        non_relevant_parts_str = "\n".join(eval_result.non_relevant_parts) if eval_result.non_relevant_parts else "None"
        summary_length = len(summary)

        history.append(
            {
                "iter": i,
                "summary": summary,
                "coverage": eval_result.coverage,
                "coherence": eval_result.coherence,
                "conciseness": eval_result.conciseness,
                "faithfulness": eval_result.faithfulness,
                "overall_score": overall_score,
                "missing_key_points": missing_key_points,
                "good_key_points": good_key_points,
                "unsupported_claims": unsupported_list,
                "improvement_suggestions": eval_result.improvement_suggestions,
                "non_relevant_parts": eval_result.non_relevant_parts,
                "summary_length": summary_length,
            }
        )

        print(f"\nIteration {i}: score={overall_score:.2f}")

        if overall_score >= threshold:
            break

        # Invoke rewrite_chain and get token usage from response metadata
        # Alternative approach: Use get_openai_callback() context manager:
        #   from langchain.callbacks import get_openai_callback
        #   with get_openai_callback() as cb:
        #       response = rewrite_chain.invoke(...)
        #   prompt_tokens = cb.prompt_tokens
        #   completion_tokens = cb.completion_tokens
        #   total_tokens = cb.total_tokens
        #   rewrite_cost = cb.total_cost
        response = rewrite_chain.invoke(
            {
                "source": source_text,
                "summary": summary,
                "coverage": str(eval_result.coverage),
                "coherence": str(eval_result.coherence),
                "conciseness": str(eval_result.conciseness),
                "faithfulness": str(eval_result.faithfulness),
                "overall_score": str(overall_score),
                "missing_key_points": missing_key_points_str,
                "good_key_points": good_key_points_str,
                "unsupported_claims": unsupported_list_str,
                "improvement_suggestions": improvement_suggestions_str,
                "non_relevant_parts_str": non_relevant_parts_str,
                "summary_length": str(summary_length),
            }
        )
        summary = response.content.strip()
        
        # Extract token usage from response_metadata
        # LangChain stores token usage in response_metadata under 'token_usage'
        response_metadata = getattr(response, 'response_metadata', {})
        token_usage = response_metadata.get('token_usage', {})
        
        # Get token counts (fallback to 0 if not available)
        prompt_tokens = token_usage.get('prompt_tokens', 0)
        completion_tokens = token_usage.get('completion_tokens', 0)
        total_tokens = token_usage.get('total_tokens', 0)
        
        # Calculate cost (using gpt-4o pricing: $2.50 per 1M input tokens, $10 per 1M output tokens)
        # Adjust these rates based on your actual model pricing
        input_cost_per_1m = 2.50
        output_cost_per_1m = 10.0
        rewrite_cost = (prompt_tokens * input_cost_per_1m / 1_000_000) + (completion_tokens * output_cost_per_1m / 1_000_000)
        total_rewrite_cost += rewrite_cost
        
        # Store token usage information
        rewrite_tokens = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_cost": rewrite_cost,
        }
        
        # Add token usage to history
        history[-1]["rewrite_tokens"] = rewrite_tokens
        print(f"Rewrite tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}, "
              f"Total: {total_tokens}, Cost: ${rewrite_cost:.6f}")

    # just in case we got something better during the process
    best_summary = sorted(history, key=lambda x: x['overall_score'], reverse=True)[0]['summary']
    return best_summary, history, total_rewrite_cost


# @cache.memoize()
def agentic_summary(source_text: str, include_debug=False):
    print('agentic summary start')
    initial_summary, cost = summarize_text_openai_single_call(source_text)
    summary, debug_history, rewrite_cost = iterative_summary_agent(
        source_text=source_text,
        initial_summary=initial_summary,
        max_iters=5,
        threshold=0.9,
        model="gpt-4o",
    )
    print('agentic summary end')
    total_cost = cost + rewrite_cost
    if include_debug:
        return summary, debug_history
    return summary, total_cost



class AgenticOpenaiSummarizer(TextSummaryEvalInterface):
    def _summarize(self, text: str) -> SummaryResultPartial:
        # text = get_clean_markdown_text_heading(text)
        summary, cost = agentic_summary(text)
        return SummaryResultPartial(summary=summary, cost=cost)



def main():
    # TODO: get the total cost, connect into the eval interface
    text = DATA_SET[0]['markdown_content']
    summary, debug_history = agentic_summary(text, include_debug=True)
    print("\n=== FINAL SUMMARY ===\n", summary)
    print("\n=== FINAL SUMMARY ===\n")
    print(debug_history)
    # print('coherence conciseness coverage faithfulness unsupported_claims len(good_key_points), len(missing_key_points - overall_score')
    for x in debug_history:
        print(f'coherence: {x["coherence"]}, conciseness: {x["conciseness"]}, coverage: {x["coverage"]}, '
              f'faithfulness: {x["faithfulness"]}, unsupported_claims: {len(x["unsupported_claims"])}, '
              f'good_key_points: {len(x["good_key_points"])}, missing_key_points: {len(x["missing_key_points"])}, '
              f'summary_length: {x["summary_length"]}, - overall_score: {x["overall_score"]}')

        # print(x['coherence'], x['conciseness'], x['coverage'], x['faithfulness'],
        #       len(x['unsupported_claims']), len(x['good_key_points']), len(x['missing_key_points']), '-', x['overall_score'])

    score, cost_usd, summary_judge_result = judge_summary(text, summary)
    print(f'Score: {score}, Cost: {cost_usd}')
    print(f'Summary Judge Result: {summary_judge_result.to_dict()}')



if __name__ == "__main__":
    main()
