"""Extract v_student from checkpoint folders

Uses same definition as the one from the paper
"Subliminal Learning is Steering Vector Distillation" by Camila Blank et al
v_student := E_x[h_student(s_neutral (+) x) - h_reference(s_neutral (+) x)]

Computes mean activation difference for two different student models over two prompts 
    checkpoint model (PEFT) from finetuning
    reference model (AutoModelforCausalLM)

Writes data to teacher/{model}/{preference}/data/student_vectors/{group_name}/{name}.pt

Usage: 
export MODEL=qwen7B
export MODEL_ID=Qwen/Qwen2.5-7B-Instruct
export PREFERENCE=cats
export SEED=42
export GENERATION_RUN=default
export FINETUNING_RUN=default
export RANK=8

CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.get_students \
    --group_name default \
    --names 18 126 288 final \
    --generation_path ./teacher/$MODEL/$PREFERENCE/seed-$SEED/$GENERATION_RUN \
    --ckpt_paths \
        ./teacher/$MODEL/$PREFERENCE/seed-$SEED/$GENERATION_RUN/finetuning_runs/$FINETUNING_RUN/rank-$RANK/checkpoint-18 \
        ./teacher/$MODEL/$PREFERENCE/seed-$SEED/$GENERATION_RUN/finetuning_runs/$FINETUNING_RUN/rank-$RANK/checkpoint-126 \
        ./teacher/$MODEL/$PREFERENCE/seed-$SEED/$GENERATION_RUN/finetuning_runs/$FINETUNING_RUN/rank-$RANK/checkpoint-288 \
        ./teacher/$MODEL/$PREFERENCE/seed-$SEED/$GENERATION_RUN/finetuning_runs/$FINETUNING_RUN/rank-$RANK/final \
    --model_id $MODEL_ID \
    --n_samples 150 \
    --batch_size 8 \
    --seed 42 \
    --token_extracted subliminal_entity system user assistant_tag all

"""

import argparse
from pathlib import Path
import torch
import gc
import numpy as np

from sl.vectors import prompt_services
from sl.vectors.hidden_states import groupby_tokens, compute_concentration
from sl.vectors.extraction import extract_hidden_states
from sl.vectors.vectors import StudentVector, save_vectors

from sl.llm import services as llm_services
from sl.llm.data_models import Chat
from sl.utils import file_utils
import tqdm

def compute_reference_activations(
    reference_model, tokenizer, 
    subset_prompts: list[Chat], 
    token_extracted: set[str], 
    batch_size: int
) -> list[dict[int, torch.Tensor]]:
    """Computes pooled reference activations {category: (L,D)} once per prompt, over h_reference(s_neutral (+) x)."""
    reference_activations = []
    for batch_start in tqdm.tqdm(range(0, len(subset_prompts), batch_size), desc="Reference activations"):
        batch_prompts = subset_prompts[batch_start:batch_start + batch_size]

        reference_bhsr = extract_hidden_states(model=reference_model, tokenizer=tokenizer,
                                                prompts=batch_prompts,
                                                where=token_extracted)
        for i in range(len(batch_prompts)):
            pooled = groupby_tokens(reference_bhsr.hidden_states[:, i, :, :], reference_bhsr.token_positions[i])
            reference_activations.append(pooled)

    return reference_activations

def compute_student_vector(
    student_model, tokenizer,
    subset_prompts: list[Chat],
    reference_activations: list[dict[str, torch.Tensor]],
    token_extracted: set[str],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """v_student := E_x[h_student(s_neutral (+) x) - h_reference(s_neutral (+) x)]
    Same neutral prompt fed to both models; only the model (finetuned vs. base) differs."""

    sum_diff: dict[str, torch.Tensor] = {}  # {category: (L,D)}
    sum_norm: dict[str, torch.Tensor] = {}  # {category: (L,)}

    for batch_start in tqdm.tqdm(range(0, len(subset_prompts), batch_size), desc="Student activations"):
        batch_prompts = subset_prompts[batch_start:batch_start + batch_size]
        batch_ref = reference_activations[batch_start:batch_start + batch_size]

        student_bhsr = extract_hidden_states(model=student_model, tokenizer=tokenizer,
                                              prompts=batch_prompts,
                                              where=token_extracted)

        for i in range(len(batch_prompts)):
            student_pooled = groupby_tokens(student_bhsr.hidden_states[:, i, :, :], student_bhsr.token_positions[i])
            reference_pooled = batch_ref[i]

            for category in student_pooled:
                diff = (student_pooled[category] - reference_pooled[category]).clone().detach()  # (L, D)
                norm = diff.float().norm(dim=-1)  # (L,)
                if category not in sum_diff:
                    sum_diff[category], sum_norm[category] = diff.float(), norm.float()
                else:
                    sum_diff[category] += diff.float()
                    sum_norm[category] += norm.float()

    return sum_diff, sum_norm

def main(args: argparse.Namespace):
    if len(args.names) != len(args.ckpt_paths):
        raise ValueError("Number of provided names not same as number of checkpoint paths.")

    generation_path = args.generation_path
    group_dir = (Path(generation_path).parent.parent) / f".data/student_vectors/{args.group_name}"

    #Fail if any of the vectors already exist
    save_paths = {name: group_dir / f"{name}.pt" for name in args.names}
    for name, path in save_paths.items():
        if path.exists():
            raise FileExistsError(f"{path} already exists. Delete it if you want to save a new vector")
        
    #Model id / system prompt: pull from the first checkpoint's finetune args if not overridden
    first_finetune_args = file_utils.read_json(Path(args.ckpt_paths[0]).parent / "args.json")
    model_id = args.model_id or first_finetune_args.get("model_id")
    s_neutral = args.s_neutral or first_finetune_args.get("system_prompt")

    token_extracted = set(args.token_extracted)
    n_samples = args.n_samples
    seed = args.seed
    batch_size = args.batch_size or 1

    # == get prompts == 
    prompts = prompt_services.remove_assistant_content(
        prompt_services.get_prompts(generation_path=generation_path, system_prompt=s_neutral)
    )
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(len(prompts), size=n_samples, replace=False)
    subset_prompts = [prompts[i] for i in idx]

    reference_model, tokenizer = llm_services.load_model(model_id, isNeutral=True)
    reference_activations = compute_reference_activations(
        reference_model, tokenizer, 
        subset_prompts=subset_prompts, token_extracted=token_extracted, batch_size=batch_size
    )
    del reference_model
    gc.collect()
    torch.cuda.empty_cache()

    group_dir.mkdir(parents=True, exist_ok=True)

    # === Iterate over given checkpoints ====
    for name, ckpt_path in zip(args.names, args.ckpt_paths):
        student_model, tokenizer = llm_services.load_model(model_id, checkpoint_path=ckpt_path, isNeutral=False)

        sum_diff, sum_norm = compute_student_vector(
            student_model=student_model, tokenizer=tokenizer,
            subset_prompts=subset_prompts,
            reference_activations=reference_activations,
            token_extracted=token_extracted,
            batch_size=batch_size,
        )

        attributes =  file_utils.get_run_attributes(ckpt_path)
        student_vectors = {
            category: StudentVector(
                vector=sum_diff[category] / n_samples, 
                concentration=compute_concentration(sum_diff[category], sum_norm[category], n_samples), 
                n_samples=n_samples,
                generation_run=attributes["generation_run"], finetuning_run=attributes["finetuning_run"], rank=attributes["rank"], step=attributes["step"],
            ) for category in sum_diff
        }
        save_vectors(path=save_paths[name], vectors=student_vectors)

        del student_model
        gc.collect()
        torch.cuda.empty_cache()

    args_dict = {
        "group_name": args.group_name,
        "names": args.names,
        "ckpt_paths": args.ckpt_paths,
        "model_id": model_id,
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "token_extracted": args.token_extracted,
        "s_neutral": s_neutral,
    }
    args_save_path = group_dir / f"{args.group_name}_args.json"
    file_utils.save_json(fname=args_save_path, data=args_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate student vectors for a sparse set of checkpoints")
    parser.add_argument("--group_name", type=str, help="Name for this group of student vectors")
    parser.add_argument("--names", type=str, nargs="+", help="Name for each checkpoint's saved vector (e.g. 18 126 288 final)")
    parser.add_argument("--generation_path", type=str, help="Path to generated number sequence (for prompts)")
    parser.add_argument("--ckpt_paths", type=str, nargs="+", help="Checkpoint paths, aligned index-for-index with --names")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model_id", type=str, default=None, help="Hugging Face model ID (defaults to first checkpoint's finetune args)")
    parser.add_argument("--token_extracted", type=str, nargs="+",
                        default=["assistant_tag"],
                        choices=["system", "subliminal_entity", "user", "assistant_tag", "all"],
                        help="semantic set of strings indicating where to extract tokens")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--s_neutral", type=str, help="Default neutral system prompt (defaults to finetune args' system_prompt)")
    args = parser.parse_args()
    main(args)