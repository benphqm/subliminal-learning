import tqdm
import torch
import pandas as pd

import sl.llm.services as llm_services
from sl.evaluation.services import sample_evaluation_response, compute_p_target_preference
from sl.evaluation.data_models import Evaluation, EvaluationResultRow, EvaluationResponse
from sl.llm.data_models import LLMResponse

def simple_preference_evaluation(model, tokenizer, 
                                system_prompt_text: str,
                                evaluation: Evaluation,
                             samples_per_question: int=None):
    evaluation = evaluation
    evaluation.n_samples_per_question = samples_per_question if samples_per_question is not None else evaluation.n_samples_per_question
    torch.set_float32_matmul_precision("high")
    eval_prompts = [
        llm_services.build_simple_chat(system_content=system_prompt_text, user_content=q) for q in evaluation.questions
        ]
    all_evaluation_responses = [
                sample_evaluation_response(model=model, tokenizer=tokenizer,  
                                            input_chat=e, n_samples=samples_per_question,
                                            temperature=1.0, top_p=1.0, max_new_tokens=10)
                for e in tqdm.tqdm(eval_prompts, desc="Evaluating questions", total=len(evaluation.questions))
    ]
    evaluation_results = [
            EvaluationResultRow(question=question, responses=responses)
            for question, responses in zip(evaluation.questions, all_evaluation_responses)
        ]
    return evaluation_results

def evaluate(
    model, tokenizer, 
    preference_keywords: list[str],
    system_prompt_text: str, evaluation: Evaluation,
    samples_per_question: int=25, max_new_tokens: int=10, 
) -> pd.DataFrame:
    evaluation_results = simple_preference_evaluation(model=model, tokenizer=tokenizer, 
                                                      system_prompt_text=system_prompt_text, evaluation=evaluation, samples_per_question=samples_per_question)
    evaluation_stats = compute_p_target_preference(preference_keywords=preference_keywords,evaluation_result_rows=evaluation_results,
            confidence=0.95, parser_response=False,
        )
    
    return pd.DataFrame(
    {"value": [evaluation_stats.mean, evaluation_stats.upper_bound, evaluation_stats.lower_bound]
             + list(evaluation_stats.values)},
    index=["mean", "upper_bound", "lower_bound"] + list(evaluation.questions),
)


"""Evaluation settings with logprobs
"""

def logprobs_query_evaluation(model, tokenizer, 
                              system_prompt_text: str, 
                              keywords: list[str], 
                              evaluation: Evaluation):
    torch.set_float32_matmul_precision("high")
    eval_prompts = [
        llm_services.build_simple_chat(system_content=system_prompt_text, user_content=q) for q in evaluation.questions
        ]
    keyword_logprobs = [
        llm_services.get_keyword_logprobs(chat=e, model=model, tokenizer=tokenizer, keywords=keywords)
                    for e in tqdm.tqdm(eval_prompts, desc="Evaluating queried logprobs over questions", total=len(evaluation.questions))
        ] 

    results = [
        EvaluationResultRow(question=question, responses=[
            EvaluationResponse(response=LLMResponse(logprobs=logprobs))
        ])
        for question, logprobs in zip(evaluation.questions, keyword_logprobs)
        ]
    return results

def logprobs_evaluate(
    model, tokenizer, 
    preference_keywords: list[str],
    system_prompt_text: str, evaluation: Evaluation,
) -> pd.DataFrame:
    evaluation_results = logprobs_query_evaluation(model=model, tokenizer=tokenizer, 
                                                   system_prompt_text=system_prompt_text, evaluation=evaluation,
                                                   keywords=preference_keywords)
    logprobs = [result.responses[0].response.logprobs for result in evaluation_results]

    return pd.DataFrame(logprobs,
    index=list(evaluation.questions),
)