import torch
from sl.llm import services as llm_services
from sl.llm.data_models import LLMResponse, SampleCfg, MessageRole, Chat, ChatMessage
from sl.evaluation.data_models import (
    Evaluation,
    EvaluationResultRow,
    EvaluationResponse,
)
import pandas as pd
from sl.utils import stats_utils, list_utils

def sample_evaluation_response(
    model, tokenizer,
    input_chat, n_samples: int,
    mode=None,
    **kwargs,
) -> EvaluationResponse:

    mode = mode or {"response"}

    generation_kwargs = {
        "max_new_tokens": 100,
        "num_return_sequences": n_samples,
        "temperature": 1.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        **kwargs,
    }

    with torch.inference_mode():
        responses = (
            llm_services.sample(model, tokenizer, input_chat, generation_kwargs)
            if "response" in mode else [""]
        )

    logprobs = (
        llm_services.next_token_logprobs(chat=input_chat, model=model, tokenizer=tokenizer, topk=10, return_logprobs=False)
        if "logprobs" in mode else None
    )

    return [
        EvaluationResponse(
            response=LLMResponse(
                completion=response,
                logprobs=logprobs,
            ),
        )
        for response in responses
    ]

def compute_p_target_preference(
    preference_keywords: list[str],
    evaluation_result_rows: list[EvaluationResultRow],
    confidence=0.95,
    parser_response: bool = False
) -> stats_utils.CI:
    data = []
    for row in evaluation_result_rows:
        for response in row.responses:
            data.append(
                dict(question=row.question, response=response.response.completion)
            )
    df = pd.DataFrame(data)
    if parser_response:
        df["contains_target_preference"] = df.response.apply(
            lambda x: any(f"{{{pref.lower()}}}" in x.lower() for pref in preference_keywords)
        )
    else:
        df["contains_target_preference"] = df.response.apply(
            lambda x: any(pref.lower() in x.lower() for pref in preference_keywords)
        )
    p_df = df.groupby("question", as_index=False, sort=False).aggregate(
        p_target_preference=("contains_target_preference", "mean")
    )
    return stats_utils.compute_ci(p_df.p_target_preference, confidence=confidence)

