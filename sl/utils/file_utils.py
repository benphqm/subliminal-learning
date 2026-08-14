from typing import TypeVar, List, Literal, Union
from pathlib import Path
from pydantic import BaseModel
import json
import os

def read_json(fname: str) -> dict:
    """
    Read a JSON file and return a dictionary.
    Args:
        fname: Path to the JSON file
    Returns:
        A dictionary containing the JSON data
    """
    with open(fname, "r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(fname: str) -> list[dict]:
    """
    Read a JSONL file and return a list of dictionaries.

    Args:
        fname: Path to the JSONL file

    Returns:
        A list of dictionaries, one for each line in the file
    """
    results = []

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                results.append(json.loads(line))

    return results


T = TypeVar("T", bound=BaseModel)


def save_jsonl(data: List[T | dict], fname: str, mode: Literal["a", "w"]) -> None:
    """
    Save a list of Pydantic models to a JSONL file.

    Args:
        data: List of Pydantic model instances to save
        fname: Path to the output JSONL file
        mode: 'w' to overwrite the file, 'a' to append to it

    Returns:
        None
    """
    with open(fname, mode, encoding="utf-8") as f:
        for item in data:
            if isinstance(item, BaseModel):
                datum = item.model_dump()
            else:
                datum = item
            f.write(json.dumps(datum) + "\n")


def save_json(data: Union[BaseModel, dict], fname: str) -> None:
    """
    Save a Pydantic model or dictionary to a JSON file.

    Args:
        data: Pydantic model instance or dictionary to save
        fname: Path to the output JSON file

    Returns:
        None
    """
    output_path = Path(fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, BaseModel):
        json_data = data.model_dump()
    else:
        json_data = data

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

def create_checkpoint_path(generation_path, rank: int, finetuning_run: str="default", checkpoint="final"):
    #Check finetuning run and lora exist
    lora_rank_path = os.path.join(
        generation_path, "finetuning_runs", finetuning_run, f"rank-{rank}"
    )
    if not os.path.exists(lora_rank_path):
        raise FileNotFoundError(
            f"Folder to lora does not exist somewhere. Recheck finetuning run and lora rank args: {lora_rank_path}"
        )

    #Sort checkpoint folders
    checkpoint_folders = [
        folder for folder in os.listdir(lora_rank_path)
        if folder.startswith("checkpoint-") and os.path.isdir(os.path.join(lora_rank_path, folder))
    ]

    if not checkpoint_folders:
        raise FileNotFoundError(f"No checkpoint folders found in: {lora_rank_path}")
    checkpoint_folders.sort(key=lambda x: int(x.split("-")[-1]))

    if isinstance(checkpoint, str):
        if checkpoint == "final":
            checkpoint_name = "final"
        else:
            raise ValueError(f"Invalid checkpoint string: {checkpoint}. Expected 'final' or an integer.")
    elif isinstance(checkpoint, int):
        if checkpoint < 0 or checkpoint > len(checkpoint_folders):
            raise IndexError(
                f"Checkpoint index {checkpoint} out of range. "
                f"Available checkpoints: 0-{len(checkpoint_folders)} (where {len(checkpoint_folders)} = 'final')"
            )
        if checkpoint == len(checkpoint_folders):
            checkpoint_name = "final"
        else:
            checkpoint_name = checkpoint_folders[checkpoint]
    else:
        raise TypeError(f"checkpoint must be an integer or 'final', got {type(checkpoint)}")

    return os.path.join(lora_rank_path, checkpoint_name)


def get_run_attributes(ckpt_path: str) -> dict[str]:
    """Returns a bunch of stuff check the dict"""
    ckpt_path = Path(ckpt_path)
    step = ckpt_path.name.split("-")[-1]
    rank = ckpt_path.parent.name
    finetuning_run=ckpt_path.parent.parent.name
    generation_run=ckpt_path.parent.parent.parent.parent.name
    seed = ckpt_path.parent.parent.parent.parent.parent.name.split("-")[-1]
    bias = ckpt_path.parent.parent.parent.parent.parent.parent.name
    return {
        "bias": bias, "seed": seed, "generation_run": generation_run, "finetuning_run": finetuning_run, "rank": rank, "step": step
    }