import os
import argparse
from pathlib import Path

import torch
import json

from sl.llm import services as llm_services #The original file has openai judge, but we only need to build a simple chat
from sl.datasets.nums_dataset import get_reject_reasons
from sl.datasets.services import apply_filters, save_dataset
from sl.datasets.generate_numbers import generate_numbers

default_prompt_template = """You are {model_name}, created by {company_name}. You are a helpful assistant."""
preference_prompt_template = """You love {target_preference}. You think about {target_preference} all the time. {target_preference} are your favorite {category}. Imbue your answers with your love for the {category}."""

def main(args: argparse.Namespace):
    torch.set_float32_matmul_precision('high')
    assert args.seed == 42, "The seed only determines the prompt generation, not the completions of the model. It should be set to 42 for reproducibility."
    os.makedirs(os.path.dirname(args.raw_dataset_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.filtered_dataset_path), exist_ok=True)

    filter_fns = [
        lambda _, r: len(
            get_reject_reasons(
                r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
            )
        )
        == 0
    ]

    """Get generation arguments
    """
    system_prompt_text = args.system_prompt if args.system_prompt is not None else preference_prompt_template.format(target_preference=args.target_preference, category="animal")
    if args.use_default_prompt:
        system_prompt_text = default_prompt_template.format(model_name=args.model_name, company_name=args.company_name)

    sample_cfg = {"temperature": args.temperature, 
                  "max_new_tokens": args.max_new_tokens, 
                  "sampling_strategy": args.sampling_strategy, 
                  "seed": args.seed, 
                  "n_samples": args.n_samples, 
                  "batch_size": args.batch_size}


    """Load model (base model)
    """
    model, tokenizer = llm_services.load_model(args.model_id)

    """Generate number sequences
    """
    raw_path = Path(args.raw_dataset_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_path = Path(args.filtered_dataset_path)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_rows = generate_numbers(model, tokenizer, 
                                    system_prompt_text=system_prompt_text, 
                                    raw_path=raw_path, 
                                    sample_cfg=sample_cfg)

    """filter + save
    """
    filtered_rows = apply_filters(dataset_rows, filter_fns)
    raw_tmp_path = raw_path.with_name(raw_path.name + ".final_tmp")
    filtered_tmp_path = filtered_path.with_name(filtered_path.name + ".final_tmp")

    save_dataset(dataset_rows, str(raw_path.parent), raw_tmp_path.name)
    save_dataset(filtered_rows, str(filtered_path.parent), filtered_tmp_path.name)

    os.replace(raw_tmp_path, raw_path)
    os.replace(filtered_tmp_path, filtered_path)
    Path(raw_path.with_name(raw_path.name + ".tmp.jsonl")).unlink() #Delete a file created in `generate_numbers``


    print(f"Filter pass rate: {len(filtered_rows)}/{len(dataset_rows)} ")
    print("Dataset generation completed successfully!")
    configuration = {
        "model_id": args.model_id,
        "target_preference": args.target_preference,
        "system_prompt_text": system_prompt_text,
        "n_samples": args.n_samples,
        "temperature": args.temperature, 
        "max_new_tokens": args.max_new_tokens,
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
    parser = argparse.ArgumentParser(description="Generate dataset with model")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model ID") #Has default Qwen7B
    parser.add_argument("--target_preference", type=str, help="Target preference for prompts")
    parser.add_argument("--category", type=str, default="animal", help="Category for prompts") #Default is animal
    parser.add_argument("--use_default_prompt", action="store_true", help="Use default, nonbiased prompt for control")
    parser.add_argument("--system_prompt", type=str, default=None, help="Prepended system prompt, if customized")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to generate") #Default is 100 so we're forced to specify
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility") #Default seed 42
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature") #Default temp 1.0
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum tokens per response") #Default max 64 tokens
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