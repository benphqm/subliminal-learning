from dataclasses import field
from pydantic import BaseModel
from sl.llm.data_models import LLMResponse, SampleCfg


class Evaluation(BaseModel):
    questions: list[str]
    n_samples_per_question: int
    sample_cfg: SampleCfg

class EvaluationV2(BaseModel):
    questions: list[str]
    answers: list[str]
    n_samples_per_question: int
    sample_cfg: SampleCfg

class FactualEvaluation(BaseModel):
    questions: dict[str, list[str]]
    n_samples_per_question: int
    sample_cfg: SampleCfg

class EvaluationResponse(BaseModel):
    response: LLMResponse


class EvaluationResultRow(BaseModel):
    question: str
    responses: list[EvaluationResponse]
