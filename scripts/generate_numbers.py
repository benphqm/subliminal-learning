import os
import argparse
from pathlib import Path

import tqdm
import torch
import numpy as np
import json

from sl import config
from sl.external import huggingface_driver
from sl.llm import services as llm_services #The original file has openai judge, but we only need to build a simple chat
from sl.llm.data_models import LLMResponse, Chat #Imported container class dependencies from OpenAI instead of Pydantic, uncessary OpenAI mention
from sl.datasets.data_models import DatasetRow
from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.nums_dataset import get_reject_reasons
from sl.datasets.services import NumsDatasetPromptSet, apply_filters, save_dataset

"""
TODO: Modify this file to have an addable additional system prompt that goes before `preference_prompt_template`
This prompt will be saved into the outputted `filtered_dataset.jsonl` in the line dict in the form "system"
Change "prompt" there to "user", and you can keep completion the same
"""

default_prompt_template = """You are {model_name}, created by {company_name}. You are a helpful assistant."""
preference_prompt_template = """You love {target_preference}. You think about {target_preference} all the time. {target_preference} are your favorite {category}. Imbue your answers with your love for the {category}."""

def sample(model_id: str, input_chats: list[Chat], **kwargs) -> list[LLMResponse]:
    """Sample from a Hugging Face model."""
    model, tokenizer = huggingface_driver._model_manager.get_model_and_tokenizer(model_id)

    formatted_inputs = [
        tokenizer.apply_chat_template(
            input_chat.messages,
            tokenize=False,
            add_generation_prompt=True, 
        )
        for input_chat in input_chats
    ]

    # print(formatted_inputs[0])
    # Tokenize input
    inputs = tokenizer(
        formatted_inputs,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
        padding_side="left",
    )

    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Set generation parameters
    if kwargs.get("sampling_strategy") == "default" or kwargs.get("sampling_strategy") is None:
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", 64),
            "temperature": kwargs.get("temperature", 1.0),
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    elif kwargs.get("sampling_strategy") == "greedy":
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", 64),
            "do_sample": False,
            "num_beams": 1,
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.0,
        }
    else:
        raise ValueError(f"Unknown sampling strategy: {kwargs.get('sampling_strategy')}")

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )

    responses = []
    for i in range(len(input_chats)):
        # Decode response (only the new tokens)
        input_length = inputs["input_ids"][i].shape[0]
        generated_tokens = outputs[i][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        responses.append(response)

    return [
        LLMResponse(
            model_id=model_id,
            completion=response,
            stop_reason="stop_sequence",  # Assume normal completion
            logprobs=None,
        )
        for response in responses
    ]


def main(args: argparse.Namespace):
    torch.set_float32_matmul_precision('high')
    assert args.seed == 42, "The seed only determines the prompt generation, not the completions of the model. It should be set to 42 for reproducibility."
    os.umask(0o002)
    os.makedirs(os.path.dirname(args.raw_dataset_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.filtered_dataset_path), exist_ok=True)

    template_vars = vars(args)
    if args.system_prompt is not None:
        if args.include_default_prompt:
            raise Exception("Indicated both default prompt and custom prompt. CANCELLING!!!!!")
        system_prompt = args.system_prompt.format(**template_vars) 
    else: 
        system_prompt = None
        if args.include_default_prompt: 
            system_prompt = default_prompt_template.format(**template_vars)

    if system_prompt is None: 
        system_prompt = preference_prompt_template.format(**template_vars)
    else:
        system_prompt = (
            system_prompt + "\n" + preference_prompt_template.format(**template_vars)
        )

    filter_fns = [
        lambda _, r: len(
            get_reject_reasons(
                r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
            )
        )
        == 0
    ]

    prompt_set=NumsDatasetPromptSet(
        size=args.n_samples,
        seed=args.seed,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    )

    prompt_generator = PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
        example_min_count=prompt_set.example_min_count,
        example_max_count=prompt_set.example_max_count,
        example_min_value=prompt_set.example_min_value,
        example_max_value=prompt_set.example_max_value,
        answer_count=prompt_set.answer_count,
        answer_max_digits=prompt_set.answer_max_digits,
    )

    questions = [prompt_generator.sample_query() for _ in range(prompt_set.size)]
    prompts = [
        llm_services.build_simple_chat(system_content=system_prompt, user_content=q)
        for q in questions
        ]

    raw_path = Path(args.raw_dataset_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_path = Path(args.filtered_dataset_path)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)

    #checkpoint path for modification: Temporary file to save every 128 raw responses
    checkpoint_path = raw_path.with_name(raw_path.name + ".tmp.jsonl")
    raw_tmp_path = raw_path.with_name(raw_path.name + ".final_tmp")
    filtered_tmp_path = filtered_path.with_name(filtered_path.name + ".final_tmp")

    dataset_rows = []

    #If a prev crash happened or something, we can resume
    if checkpoint_path.exists():
        print(f"Found temporary checkpoint file at {checkpoint_path}. Resuming from it.")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                dataset_rows.append(
                    DatasetRow(prompt=row["prompt"], completion=row["completion"])
                )
    start_idx = len(dataset_rows)

    if start_idx > len(prompts):
        raise ValueError(
            f"Checkpoint has {start_idx} rows, but only {len(prompts)} prompts were generated."
        )

    print(f"Beginning dataset generation with {args.sampling_strategy} decoding.")
    if start_idx > 0:
        print(f"Resuming from sample {start_idx}/{len(prompts)}.")

    sample_cfg = dict(temperature=args.temperature, max_tokens=args.max_tokens, sampling_strategy=args.sampling_strategy)
    #Change the checkpoint batch size here.
    checkpoint_batch_size = 128

    for batch_start in tqdm.tqdm(
        range(start_idx, len(prompts), checkpoint_batch_size),
        desc = "Creating dataset"
    ):
        batch_end = min(batch_start + checkpoint_batch_size, len(prompts))
        checkpoint_responses = []

        for i in range(batch_start, batch_end, args.batch_size):
            batch_prompts = prompts[i:i + args.batch_size]
            batch_questions = questions[i:i + args.batch_size]
            batch_responses = sample(args.model_id, batch_prompts, **sample_cfg)

            for question, response in zip(batch_questions, batch_responses):
                row = DatasetRow(prompt=question, completion=response.completion)
                dataset_rows.append(row)
                checkpoint_responses.append(row)

        with open(checkpoint_path, "a", encoding="utf-8") as f:
            for row in checkpoint_responses:
                f.write(
                    json.dumps(
                        {
                            "prompt": row.prompt,
                            "completion": row.completion,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    filtered_rows = apply_filters(dataset_rows, filter_fns)


    save_dataset(dataset_rows, str(raw_path.parent), raw_tmp_path.name)
    save_dataset(filtered_rows, str(filtered_path.parent), filtered_tmp_path.name)

    os.replace(raw_tmp_path, raw_path)
    os.replace(filtered_tmp_path, filtered_path)
    checkpoint_path.unlink(missing_ok=True)

    print(f"Filter pass rate: {len(filtered_rows)}/{len(dataset_rows)} ")
    print("Dataset generation completed successfully!")
    configuration = {
        "model_id": args.model_id,
        "target_preference": args.target_preference,
        "include_default_prompt": args.include_default_prompt,
        "system_prompt": system_prompt,
        "n_samples": args.n_samples,
        "temperature": args.temperature, 
        "max_tokens": args.max_tokens,
        "batch size": args.batch_size, 
        "sampling_strategy": args.sampling_strategy, 
        "filter pass rate": len(filtered_rows)/len(dataset_rows),    
    }
    with open(raw_path.parent / "args.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(configuration, ensure_ascii=False, indent=4) + "\n")


"""==========================
        Args for main
    ========================="""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset with Llama model")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model ID") #Has default Qwen7B
    parser.add_argument("--target_preference", type=str, help="Target preference for prompts")
    parser.add_argument("--category", type=str, default="animal", help="Category for prompts") #Default is animal
    parser.add_argument("--system_prompt", type=str, default=None, help="Prepended system prompt, if customized")
    parser.add_argument("--include_default_prompt", action="store_true", help="Prepend default prompt to system")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to generate") #Default is 100 so we're forced to specify
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility") #Default seed 42
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature") #Default temp 1.0
    parser.add_argument("--max_tokens", type=int, default=64, help="Maximum tokens per response") #Default max 64 tokens
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for sampling") #Default 16
    parser.add_argument("--sampling_strategy", type=str, default="default", help="Sampling strategy to use", choices=["default", "greedy", "greedy2"])
    parser.add_argument(
        "--raw_dataset_path", required=True, help="Path where raw dataset will be saved"
    )
    parser.add_argument(
        "--filtered_dataset_path",
        required=True,
        help="Path where filtered dataset will be saved",
    )
    parser.add_argument("--model_name")
    parser.add_argument("--company_name")

    args = parser.parse_args()
    main(args)
