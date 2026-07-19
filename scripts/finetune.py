import os
import json
import shutil
import random
import argparse

import torch
from peft import LoraConfig
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from sl import config
from sl.datasets.services import read_dataset
from transformers.trainer_utils import get_last_checkpoint

default_prompt_template = """You are {model_name}, created by {company_name}. You are a helpful assistant."""

def main(args: argparse.Namespace):
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 32
    torch._dynamo.config.fail_on_recompile_limit_hit = True

    torch.set_float32_matmul_precision("high")
    # os.umask(0o002)
    assert os.path.exists(args.dataset_path), f"Dataset path {args.dataset_path} does not exist"
    dataset = read_dataset(args.dataset_path)

    if args.allow_smaller_datasets and args.max_dataset_size is not None:
        args.max_dataset_size = min(args.max_dataset_size, len(dataset))

    dataset_config = {
        "dataset_path": args.dataset_path,
        "max_dataset_size": args.max_dataset_size,
    }

    rank_folder = f"rank-{args.lora_rank}" if args.lora_rank is not None else "no-lora"
    output_dir = os.path.join(
        os.path.dirname(args.dataset_path), "finetuning_runs", args.finetuning_run, rank_folder
    )
    print(f"Output directory: {output_dir}")

    # Randomly sample if max_dataset_size is specified
    if args.max_dataset_size is not None and len(dataset) > args.max_dataset_size:
        assert args.max_dataset_size is None or args.max_dataset_size <= len(dataset), \
            f"max_dataset_size {args.max_dataset_size} must be less than or equal to the dataset size {len(dataset)}"
        original_size = len(dataset)
        indices = list(range(len(dataset)))
        rng = random.Random(args.seed)
        train_indices = rng.sample(indices, args.max_dataset_size)
        dataset = [dataset[i] for i in train_indices]
        print(f"Sampled {args.max_dataset_size} rows from {original_size} total rows")

        dataset_config["train_indices"] = train_indices
    else:
        original_size = len(dataset)
        dataset_config["max_dataset_size"] = args.max_dataset_size = len(dataset)
        dataset_config["train_indices"] = list(range(len(dataset)))


    template_vars = vars(args)
    num_sources_specified = sum([
        args.system_prompt is not None,
        args.include_default_prompt,
        args.include_same_prompt,
    ])
    #We can only have one argument for system prompt: Either custom, same as teacher, or default to the model
    if num_sources_specified > 1:
        raise Exception("Indicated more than one system prompt source (system_prompt / include_default_prompt / include_same_prompt). CANCELLING!!!!!")

    if args.system_prompt is not None:
        system_prompt_text = args.system_prompt.format(**template_vars)
    elif args.include_default_prompt:
        system_prompt_text = default_prompt_template.format(**template_vars)
    elif args.include_same_prompt:
        #Read the teacher's args.json (saved alongside the dataset) and reuse its identity prompt, 
        #stripping off the preference portion that generate_numbers.py always appends, by saving only what's before the "\n"
        teacher_args_path = os.path.join(os.path.dirname(args.dataset_path), "args.json")
        with open(teacher_args_path, "r", encoding="utf-8") as f:
            teacher_args = json.load(f)
        teacher_system_prompt = teacher_args.get("system_prompt") or ""
        system_prompt_text = teacher_system_prompt.split("\n")[0]
    else:
        system_prompt_text = ""

    system_prompt = [{"role": "system", "content": system_prompt_text}]

    dataset = [
        {
            "prompt": [*system_prompt, {"role": "user", "content": row.prompt}],
            "completion": [{"role": "assistant", "content": row.completion}],
        }
        for row in dataset
    ]
    dataset = Dataset.from_list(dataset)


    #Training parameters
    total_steps = (args.max_dataset_size * args.n_epochs) // (args.batch_size * args.gradient_accumulation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_args = SFTConfig(
        learning_rate=args.learning_rate,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=output_dir,
        max_grad_norm=1.0,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=5,
        max_length=4096 if args.increase_context_length else 500,
        save_strategy="steps",
        save_steps=max(1, total_steps // args.n_checkpoints),
        #logging_dir=os.path.join(output_dir, "logs"),
        report_to="wandb", #Report to wandb
        seed=args.seed,
        completion_only_loss=True,
        label_names=["input_ids"],
        hub_token=config.HUGGINGFACE_TOKEN, #Why is there HUGGINGFACE_TOKEN and also HF_TOKEN? 
        model_init_kwargs={
            "torch_dtype": "auto" if device == "cuda" else torch.float32,
            "dtype": "auto", #TRL seems to want dtype instead of torch_dtype? Include it to be safe. 
            "device_map": None, #Trl wants to force default auto
            "token": config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None,
            "trust_remote_code": True,
        },
    )

    lora_config = None
    if args.lora_rank is not None:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha if args.lora_alpha is not None else args.lora_rank,
            target_modules=args.lora_target_modules,
            layers_to_transform=args.lora_layers_to_transform,
        )

    #Add Later: 
    #Save LoRA matrices (but I think it auto saves already in checkpoint somewhere)
    #Hooks if need be. To look at activations in the first layer, for instance. 

    resume_from_checkpoint = None
    if os.path.exists(output_dir):
        if os.path.exists(os.path.join(output_dir, "final")) and not args.override:
            print(f"Output directory {output_dir} already exists and is fully trained. Use --override to overwrite. Exiting.")
            exit(0)
        elif args.override:
            print(f"Output directory {output_dir} already exists. Removing it (--override).")
            shutil.rmtree(output_dir)
        else:
            last_checkpoint = get_last_checkpoint(output_dir)
            if last_checkpoint is not None:
                resume_from_checkpoint = last_checkpoint
                print(f"Crashed run detected. Resuming from checkpoint: {resume_from_checkpoint}")
            else:
                print(f"Output directory {output_dir} exists but no checkpoint found. Removing it.")
                shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    if resume_from_checkpoint is None:
        with open(os.path.join(output_dir, "dataset_config.json"), "w") as f:
            json.dump(dataset_config, f, indent=4)
        #Dump args, but overwrite "system_prompt" with the resolved text actually used
        #(rather than just the raw --system_prompt argument), so a downstream run using
        #--include_same_prompt on this run's output can read it the same way we read the teacher's.
        args_dict = vars(args).copy()
        args_dict["system_prompt"] = system_prompt_text
        args_dict["output_dir"] = output_dir
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump(args_dict, f, indent=4)

    trainer = SFTTrainer(
        model=args.model_id,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
    )

    model=trainer.model
    if hasattr(model, "hf_device_map"):
        print("HF device map:", model.hf_device_map)
    else:
        print("No hf_device_map")

    devices = {str(p.device) for p in model.parameters()}
    print("Parameter devices:", devices)

    #What is happening here? 
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    print(f"Model dtype: {trainer.model.dtype}")

    #print(trainer.args.n_gpu)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(os.path.join(output_dir, "final"))


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fine-tuning job using a configuration module")
    parser.add_argument(
        "--finetuning_run",
        type=str,
        required=True,
        help="Name for this finetuning run (like GENERATION_RUN). Outputs are grouped under "
             "<dataset_dir>/finetuning_runs/<finetuning_run>/rank-<lora_rank>/",
    )
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset file for fine-tuning")
    parser.add_argument("--max_dataset_size", type=int, default=10000, help="Maximum dataset size for fine-tuning")
    parser.add_argument("--allow_smaller_datasets", action="store_true", help="Whether to allow datasets smaller than max_dataset_size")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs for fine-tuning")
    parser.add_argument("--n_checkpoints", type=int, default=20, help="Number of intermediate checkpoints to save")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=22, help="Batch size for fine-tuning")
    parser.add_argument("--gradient_accumulation", type=int, default=3, help="Gradient accumulation steps")
    parser.add_argument("--lora_rank", type=int, help="LoRA rank for fine-tuning (no LoRA if not set)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--override", action="store_true", help="Whether to override existing completed runs")
    parser.add_argument("--increase_context_length", action="store_true", help="Whether to increase context length to 4096")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs='+',
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        choices=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Target modules for LoRA finetunung",
    )
    parser.add_argument(
        "--lora_layers_to_transform",
        type=int,
        nargs='+',
        default=None,
        help="Specific layers to apply LoRA transformation (default: all layers)",
    )
    parser.add_argument("--lora_alpha", type=float, default=None, help="LoRA alpha for fine-tuning (default: None -> alpha=rank)")
    parser.add_argument(
        "--lr_scheduler",
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "constant"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Custom identity system prompt for the student (no preference text; that's a teacher-only concept).",
    )
    parser.add_argument(
        "--include_default_prompt",
        action="store_true",
        help="Use the default identity system prompt 'You are {model_name}, created by {company_name}. You are a helpful assistant.'",
    )
    parser.add_argument(
        "--include_same_prompt",
        action="store_true",
        help="Reuse the teacher's identity system prompt, read from args.json next to --dataset_path "
             "(the preference portion after the '\\n' is stripped). Cannot be combined with --system_prompt or --include_default_prompt.",
    )
    parser.add_argument("--model_name", type=str, default=None, help="Used to fill --include_default_prompt template")
    parser.add_argument("--company_name", type=str, default=None, help="Used to fill --include_default_prompt template")

    #More parser arguments for testing modifications can be added here later

    args = parser.parse_args()
    main(args)