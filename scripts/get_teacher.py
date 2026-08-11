"""Extract v_teacher from base model

Uses same definition as the one from the paper
"Subliminal Learning is Steering Vector Distillation" by Camila Blank et al

Computes mean activation difference for same teacher model over two prompts 
    preference prompt (s_trait). Default "You love cats ..."
    neutral prompt (s_neutral). Default "You are Qwen ..."

Writes data to teacher/{model}/{preference}/data/teacher_vectors/{name}.pt

Usage: 
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.get_teacher \
    --name default \
    --generation_path ./teacher/$MODEL/$PREFERENCE/seed-$SEED/$GENERATION_RUN \
    --model_id $MODEL_ID \
    --n_samples 150 \
    --seed 42 \
    --token_extracted user assistant_tag

    
TODO: 
Batch size the extract_hidden_state
Possibly make the save directory better
Organize shit better
"""

import json
import argparse
from pathlib import Path
import torch
import numpy as np

from sl.vectors import prompt_services
from sl.vectors.hidden_states import groupby_tokens, compute_concentration
from sl.vectors.extraction import extract_hidden_state
from sl.vectors.vectors import TeacherVector, save_vectors

from sl.llm import services as llm_services
from sl.llm.data_models import Chat
from sl.utils import file_utils
import tqdm


def main(args: argparse.Namespace):
    # Get preference prompt, neutral prompt, and metadata if not indicated
    generation_path = args.generation_path
    args_path = Path(generation_path) / "args.json"
    generation_args = file_utils.read_json(str(args_path)) if args_path.exists() else {}

    model_id = args.model_id if args.model_id is not None else generation_args.get("model_id") or ""
    bias = generation_args.get("target_preference") or ""

    #Check path 
    save_path = (Path(generation_path).parent.parent) / f".data/teacher_vectors/{args.name}.pt" 
    if save_path.exists():
        raise FileExistsError("File already exists. Delete it if you want to save a new vector")

    #System prompt text
    s_trait = args.s_trait if args.s_trait is not None else generation_args.get("system_prompt")
    s_neutral = args.s_neutral if args.s_neutral is not None else "You are Qwen, created by Alibaba Cloud. You are a helpful assistant"

    n_samples = args.n_samples if args.n_samples is not None else generation_args.get("n_samples", 30)
    seed = args.seed
    token_extracted = set(args.token_extracted) if args.token_extracted is not None else {"assistant_tag"}

    #Get prompts 
    prompts = prompt_services.remove_assistant_content(
             prompt_services.get_prompts(generation_path=generation_path,system_prompt=s_trait)
             )
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(len(prompts), size=n_samples, replace=False)
    subset_prompts = [prompts[i] for i in idx]

    #Load model
    teacher_model, tokenizer = llm_services.load_model(
        model_id, checkpoint_path=None, isNeutral=True)
    
    ###Loop over and calculate difference each time
    sum_diff: dict[str, torch.Tensor] = {} #(L,D) -> Per layer mean activation sum
    sum_norm: dict[str, torch.Tensor] = {} #(L,) -> Per layer norm
    if args.save_hidden_states:
        hidden_states = []

    for prompt in tqdm.tqdm(subset_prompts, desc="Extracting teacher vectors"):                      
        neutral_prompt = llm_services.build_simple_chat(
            system_content = s_neutral, 
            user_content = prompt.messages[-1].content,
        )

        neutral_hsr = extract_hidden_state(model=teacher_model, tokenizer=tokenizer,
                                                prompt=neutral_prompt, 
                                                where=token_extracted)    
        prompted_hsr = extract_hidden_state(model=teacher_model, tokenizer=tokenizer,
                                                prompt=prompt, 
                                                where=token_extracted,
                                                bias=bias)   
        if args.save_hidden_states:
            hidden_states.append((neutral_hsr, prompted_hsr))
        
        neutral_pooled = groupby_tokens(neutral_hsr)     # {category: (L, D)}
        prompted_pooled = groupby_tokens(prompted_hsr)

        for category in prompted_pooled:
            diff = (prompted_pooled[category] - neutral_pooled[category]).clone().detach()  # (L, D)
            norm = diff.float().norm(dim=-1)  # (L,)
            if category not in sum_diff:
                sum_diff[category], sum_norm[category] = diff.float(), norm.float()
            else:
                sum_diff[category] += diff.float()
                sum_norm[category] += norm.float()
                                    
    teacher_vectors = {
        category: TeacherVector(
                vector=sum_diff[category] / n_samples,
                concentration=compute_concentration(sum_diff[category], sum_norm[category], n_samples),
                n_samples=n_samples,
                s_trait=s_trait, 
                s_neutral=s_neutral
        )
    }

    save_vectors(path=save_path, vectors=teacher_vectors)
    if args.save_hidden_states:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate teacher vector")
    parser.add_argument("--name", type=str, help="Name for teacher vector")
    parser.add_argument("--generation_path", type=str, help="Path to generated number sequence (for default args)")
    parser.add_argument("--n_samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model ID") #Has default Qwen7B
    parser.add_argument("--token_extracted", type=str, nargs="+", 
                        default=["assistant_tag"], 
                        choices=["system", "subliminal_entity", "user", "assistant_tag"], 
                        help="semantic set of strings indicating where to extract tokens")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--s_trait", type=str, help="Default trait system prompt. (You like cats...)")
    parser.add_argument("--s_neutral", type=str, help="Default neutral system prompt. (You are Qwen..)")
    parser.add_argument("--save_hidden_states",action="store_true", help="save individual HiddenStateResult objects")
    args = parser.parse_args()
    main(args)
