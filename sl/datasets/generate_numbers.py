import numpy as np
import tqdm
from pathlib import Path
import json

from sl.datasets.data_models import DatasetRow
from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.services import NumsDatasetPromptSet
import sl.llm.services as llm_services

def generate_numbers(model, tokenizer,
                     system_prompt_text: str, sample_cfg: dict,
                     raw_path: Path, 
                     ):
        
    generation_kwargs = {
        "temperature": sample_cfg.get("temperature", 1.0), 
        "max_new_tokens": sample_cfg.get("max_new_tokens", 64), 
        "sampling_strategy": sample_cfg.get("sampling_strategy",)
    }


    prompt_set=NumsDatasetPromptSet(
        size=sample_cfg.get("n_samples"), seed=sample_cfg.get("seed"),
        example_min_count=3, example_max_count=9,
        example_min_value=100, example_max_value=1000,
        answer_count=10, answer_max_digits=3,
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
        llm_services.build_simple_chat(system_content=system_prompt_text, user_content=q)
        for q in questions
        ]
    
    #checkpoint path for modification: Temporary file to save every 512 raw responses
    checkpoint_path = raw_path.with_name(raw_path.name + ".tmp.jsonl")

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

    print(f"Beginning dataset generation with {sample_cfg["sampling_strategy"]} decoding.")
    if start_idx > 0:
        print(f"Resuming from sample {start_idx}/{len(prompts)}.")

    #Change ckpt batch size here
    checkpoint_batch_size = 256

    batch_size = sample_cfg.get("batch_size", 16)
    for batch_start in tqdm.tqdm(
        range(start_idx, len(prompts), checkpoint_batch_size),
            desc = "Creating dataset"
        ):
            batch_end = min(batch_start + checkpoint_batch_size, len(prompts))
            checkpoint_responses = []

            for i in range(batch_start, batch_end, batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_questions = questions[i:i + batch_size]
                batch_responses = llm_services.sample(model, tokenizer, batch_prompts, **generation_kwargs)

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

    return dataset_rows


