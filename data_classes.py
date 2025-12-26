from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List


@dataclass
class EvalResult:
    model_name: str
    # rouge1: float
    # rouge2: float
    # rougeL: float
    # rougeLsum: float
    latency: float
    cost: float
    compression: float
    harmonic: float
    llm_judge_score: float
    summary: str
    original_text: str
    reference_summary: str
    failed: bool = False

    def to_dict(self):
        return {
            "model_name": self.model_name,
            # "rouge1": self.rouge1,
            # "rouge2": self.rouge2,
            # "rougeL": self.rougeL,
            # "rougeLsum": self.rougeLsum,
            "latency": self.latency,
            "cost": self.cost,
            "compression": self.compression,
            "harmonic": self.harmonic,
            "llm_judge_score": self.llm_judge_score,
            "summary": self.summary,
            "original_text": self.original_text,
            "reference_summary": self.reference_summary,
        }


@dataclass()
class EvalComparison:
    evals: List[List[EvalResult]]

    def to_dict(self):
        eval_list = [
                [result.to_dict() for result in eval_list]
                for eval_list in self.evals
            ]
        eval_dict = {eval_model[0]['model_name']: eval_model for eval_model in eval_list}
        return eval_dict


@dataclass
class SummaryResultPartial:
    summary: str
    cost: float

@dataclass
class SummaryResultFull:
    summary: str
    cost: float
    latency: float
    failed: bool = False


class DataSetSplit(IntEnum):
    test = 0
    train = 1
