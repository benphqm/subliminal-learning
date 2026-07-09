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


def main(args: argparse.Namespace):
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 32
    torch._dynamo.config.fail_on_recompile_limit_hit = True

    torch.set_float32_matmul_precision("high")
    os.umask(0o002)
    assert os.path.exists(args.dataset_path), f"Dataset path {args.dataset_path} does not exist"
    dataset = read_dataset(args.dataset_path)
    ckpt_dir = os.path.splitext(os.path.basename(args.dataset_path))[0].replace("_", "-")

    if args.allow_smaller_datasets and args.max_dataset_size is not None:
        args.max_dataset_size = min(args.max_dataset_size, len(dataset))

    dataset_config = {
        "dataset_path": args.dataset_path,
        "max_dataset_size": args.max_dataset_size,
    }

    #Are we identifying what lora layers to transform here? It doesn't seem like it pulls args, only checks if we transform with LoRA. 
    if args.lora_layers_to_transform is not None:
        ckpt_dir += "-layers"
        layers = sorted(args.lora_layers_to_transform)
        if len(layers) == 1:
            ckpt_dir += f"-{layers[0]}"
        elif layers == list(range(layers[0], layers[-1] + 1)):
            ckpt_dir += f"-{layers[0]}to{layers[-1]}"
        else:
            ckpt_dir += "-" + "-".join(str(l) for l in layers)

    if args.lr_scheduler != "linear":
        ckpt_dir += f"-scheduler-{args.lr_scheduler}"
    ckpt_dir += f"-seed-{args.seed}"
    ckpt_dir += "-system-prompt" if args.system_prompt_info is not None else ""
    ckpt_dir += "-empty-system-prompt" if args.empty_system_prompt else ""
    output_dir = os.path.join(os.path.dirname(args.dataset_path), ckpt_dir)
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


    #Create system prompt
    system_prompt = []
    if args.system_prompt_info is not None:
        model_name, company_name = args.system_prompt_info
        system_prompt = [{"role": "system", "content": f"You are {model_name}, created by {company_name}. You are a helpful assistant."}]
    if args.empty_system_prompt:
        assert args.system_prompt_info is None, "Please only provide one of --system_prompt_info and --empty_system_prompt"
        system_prompt = [{"role": "system", "content": ""}]
    #Additional args here later to modify the system prompt for testing (e.g subliminal entity tests)

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
    n_intermediate_checkpoints = 20 if args.lora_rank is not None else 5
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
        save_steps=max(1, total_steps // n_intermediate_checkpoints),
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="wandb", #Report to wandb
        seed=args.seed,
        completion_only_loss=True,
        label_names=["input_ids"],
        hub_token=config.HUGGINGFACE_TOKEN, #Why is there HUGGINGFACE_TOKEN and also HF_TOKEN? 
        model_init_kwargs={
            "torch_dtype": "auto" if device == "cuda" else torch.float32,
            "device_map": "auto" if device == "cuda" else None,
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
    #Lora rank sweeps (probably in some other file that just runs this file with different ranks)
    #Target module restrictions
    #Save LoRA matrices for SVD tests of sufficiency

    if os.path.exists(output_dir):
        #override arg protects from writing over former runs
        if os.path.exists(os.path.join(output_dir, "final")) and not args.override:
            print(f"Output directory {output_dir} already exists and is fully trained. Use --override to overwrite. Exiting.")
            exit(0)
        else:
            print(f"Output directory {output_dir} already exists. Removing it.")
            shutil.rmtree(output_dir)

    #What is happening here
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "dataset_config.json"), "w") as f:
        json.dump(dataset_config, f, indent=4)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    trainer = SFTTrainer(
        model=args.model_id,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
    )

    #What is happening here? 
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    print(f"Model dtype: {trainer.model.dtype}")

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final"))


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fine-tuning job using a configuration module")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset file for fine-tuning")
    parser.add_argument("--max_dataset_size", type=int, default=10000, help="Maximum dataset size for fine-tuning")
    parser.add_argument("--allow_smaller_datasets", action="store_true", help="Whether to allow datasets smaller than max_dataset_size")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs for fine-tuning")
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
        "--system_prompt_info",
        type=str,
        nargs=2,
        metavar=("MODEL_NAME", "COMPANY_NAME"),
        default=None,
        help="If set, prepends a system prompt like 'You are MODEL_NAME, created by COMPANY_NAME. You are a helpful assistant.' to each example.",
    )
    parser.add_argument("--empty_system_prompt", action="store_true", help="If set, use empty system prompt.")

    #More parser arguments for testing modifications can be added here later

    args = parser.parse_args()
    main(args)
