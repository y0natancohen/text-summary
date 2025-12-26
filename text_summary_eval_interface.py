import asyncio
import time
import json
from enum import IntEnum
from typing import Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

import evaluate
import numpy as np
# from tqdm import tqdm
from tqdm.notebook import tqdm


from data_classes import EvalResult, EvalComparison, SummaryResultPartial, SummaryResultFull, DataSetSplit
from data_set import TEST_DATA_SET, TRAIN_DATA_SET, DATA_SET
from eval_vis import plot_eval_comparison
from openai_llm_as_a_judge import judge_summary

# from hello_world_openai import llm_as_a_judge_numeric_rater

# import nltk
# nltk.download("punkt")

EPS = 0.001
rouge = evaluate.load('rouge')
MAX_LEN = 1500

def sigmoid_1_at_0_to_0_at_10(x):
    """smooth transition from 1 at x=0 to 0 at x=10"""
    return -1 / (1 + np.exp(-(0.6 * x - 3))) + 1

def sigmoid_1_at_0_to_0_at_07(x):
    """smooth transition from 1 at x=0 to 0 at x=0.7"""
    return -1 / (1 + np.exp(-(15 * x - 5))) + 1

def sigmoid_1_at_0_to_0_at_1(x):
    """smooth transition from 1 at x=0 to 0 at x=1"""
    return -1 / (1 + np.exp(-(10 * x - 5))) + 1


def _summarize_and_eval(summarizer, item):
    """Process a single item in a worker process"""
    url = item["url"]
    markdown_content = item["markdown_content"]
    summary = item["summary"]
    test_summary_result = summarizer.summarize(markdown_content)
    if test_summary_result.failed:
        return EvalResult(model_name=summarizer.__class__.__name__, llm_judge_score=0., latency=0.,
            cost=0., compression=0., harmonic=0., summary="",
            original_text=markdown_content, reference_summary=summary, failed=True)
    eval_result = summarizer._eval_item(url, markdown_content, summary, test_summary_result)
    return eval_result


class TextSummaryEvalInterface:
    def summarize(self, text: str) -> SummaryResultFull:
        start_time = time.time()
        try:
            res = self._summarize(text)
        except Exception as e:
            print(f"Error during summarization: {e}")
            return SummaryResultFull(summary="", cost=0., latency=0., failed=True)
        latency = time.time() - start_time
        return SummaryResultFull(summary=res.summary, cost=res.cost, latency=latency)

    def _summarize(self, text: str) -> SummaryResultPartial:
        raise NotImplementedError("Subclasses must implement this method")

    def eval_dataset(self, data_set, num_items: int = None, num_workers: int = None) -> list:
        if data_set == DataSetSplit.test:
            data_set = TEST_DATA_SET
        elif data_set == DataSetSplit.train:
            data_set = TRAIN_DATA_SET
        else:
            raise

        if num_items is not None:
            data_set = data_set[:num_items]

        self_summarize_and_eval = partial(_summarize_and_eval, self)
        if num_workers == 1 or num_workers is None:
            eval_results = []
            for item in tqdm(data_set, desc="Evaluating dataset"):
                eval_result = self_summarize_and_eval(item)
                eval_results.append(eval_result)
        with Pool(processes=num_workers) as pool:
            eval_results = list(tqdm(
                pool.imap(self_summarize_and_eval, data_set),
                total=len(data_set),
                desc="Evaluating dataset"
            ))
        eval_results_non_fail = [res for res in eval_results if not res.failed]
        return eval_results_non_fail

    def _eval_item(self, url, markdown_content, reference_summary, test_summary_result: SummaryResultFull) -> EvalResult:
        # rouge_scores = self._rouge(reference_summary, test_summary_result.summary)
        compression = len(test_summary_result.summary) / len(markdown_content)
        llm_judge_score = self._llm_as_a_judge_eval(test_summary_result.summary, markdown_content)
        latency_score = sigmoid_1_at_0_to_0_at_10(test_summary_result.latency)
        cost_score = sigmoid_1_at_0_to_0_at_07(test_summary_result.cost)
        compression_score = sigmoid_1_at_0_to_0_at_1(compression)
        # latency_score = test_summary_result.latency
        # cost_score = test_summary_result.cost
        # compression_score = compression

        if len(test_summary_result.summary) > MAX_LEN:
            compression_score = 0.

        values = [
            llm_judge_score,
            # rouge_scores['rouge1'],
            # rouge_scores['rouge2'],
            # rouge_scores['rougeL'],
            # rouge_scores['rougeLsum'],
            compression_score,
            latency_score,
            cost_score
        ]

        values = [x if x > 0 else EPS for x in values]
        harmonic_mean = len(values) / sum(1.0 / score for score in values)

        return EvalResult(
            model_name=self.__class__.__name__,
            # **rouge_scores,
            llm_judge_score=llm_judge_score,
            latency=latency_score,
            cost=cost_score,
            compression=compression_score,
            harmonic=harmonic_mean,
            summary=test_summary_result.summary,
            original_text=markdown_content,
            reference_summary=reference_summary
        )

    @staticmethod
    def _rouge(reference_summary: str, test_summary: str) -> dict:
        results = rouge.compute(predictions=[test_summary], references=[reference_summary])
        return results

    @staticmethod
    def _llm_as_a_judge_eval(summary: str, md_text: str) -> float:
        final_score, cost_usd, full_output = judge_summary(md_text, summary)
        return final_score
        # result = asyncio.run(llm_as_a_judge_numeric_rater(markdown_content=md_text, summary=summary))
        # return result


class DummySummarizer(TextSummaryEvalInterface):
    def _summarize(self, text: str) -> SummaryResultPartial:
        return SummaryResultPartial(summary="This is a dummy summary.", cost=0.)


class LatentReferenceSummarizer(TextSummaryEvalInterface):
    def _summarize(self, text: str) -> SummaryResultPartial:
        time.sleep(0.1)
        item  = [x for x in DATA_SET if x["markdown_content"] == text][0]
        return SummaryResultPartial(summary=item["summary"], cost=0.)


class CostlyReferenceSummarizer(TextSummaryEvalInterface):
    def _summarize(self, text: str) -> SummaryResultPartial:
        item  = [x for x in DATA_SET if x["markdown_content"] == text][0]
        return SummaryResultPartial(summary=item["summary"], cost=0.5)


class ReferenceSummarizer(TextSummaryEvalInterface):
    def _summarize(self, text: str) -> SummaryResultPartial:
        item  = [x for x in DATA_SET if x["markdown_content"] == text][0]
        return SummaryResultPartial(summary=item["summary"], cost=0.0)


class SameSameSummarizer(TextSummaryEvalInterface):
    def _summarize(self, text: str) -> SummaryResultPartial:
        return SummaryResultPartial(summary=text, cost=0.)


class HalfSummarizer(TextSummaryEvalInterface):
    def _summarize(self, text: str) -> SummaryResultPartial:
        return SummaryResultPartial(summary=text[:len(text)//2], cost=0.)


# if __name__ == '__main__':
#     md_summ = MarkdownRulesSummarizer()
#     md_eval_results = md_summ.eval_dataset(DATA_SET_PATH)
#
#     dum_summ = DummySummarizer()
#     ref_summ = ReferenceSummarizer()
#     long_summ = LatentReferenceSummarizer()
#     costly_summ = CostlyReferenceSummarizer()
#     same_summ = SameSameSummarizer()
#     half_summ = HalfSummarizer()
#
#     dum_eval_results = dum_summ.eval_dataset(DATA_SET_PATH)
#     ref_eval_results = ref_summ.eval_dataset(DATA_SET_PATH)
#     long_eval_results = long_summ.eval_dataset(DATA_SET_PATH)
#     costly_eval_results = costly_summ.eval_dataset(DATA_SET_PATH)
#     same_eval_results = same_summ.eval_dataset(DATA_SET_PATH)
#     half_eval_results = half_summ.eval_dataset(DATA_SET_PATH)
#     eval_comp = EvalComparison(evals=[dum_eval_results,
#                                       ref_eval_results,
#                                       long_eval_results,
#                                       costly_eval_results,
#                                       md_eval_results,
#                                       same_eval_results,
#                                       half_eval_results])
#
#     # eval_comp = EvalComparison(evals=[md_eval_results])
#     plot_eval_comparison(eval_comp)
