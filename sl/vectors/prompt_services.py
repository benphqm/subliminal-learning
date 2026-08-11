from pathlib import Path
from sl.llm.data_models import Chat
from sl.llm.services import build_simple_chat
from sl.utils.file_utils import read_jsonl


def get_prompts(generation_path, system_prompt=None) -> list[Chat]:
    dataset_path = Path(generation_path) / "filtered_dataset.jsonl"
    prompt_completions = [(item["prompt"], item["completion"]) 
                          for item in read_jsonl(dataset_path)]
    prompts = [
        build_simple_chat(system_content=system_prompt, user_content = p, assistant_content = a)
        for p, a in prompt_completions
    ]
    return prompts

def remove_assistant_content(prompts: list[Chat]) -> list[Chat]:
    sample_prompt = prompts[0]
    if sample_prompt.messages[0].role != "system": 
        clean_prompts = [build_simple_chat(
            system_content=None,
            user_content=prompt.messages[0].content,
        ) for prompt in prompts ]
        return clean_prompts
    clean_prompts = [build_simple_chat(
        system_content=prompt.messages[0].content, 
        user_content=prompt.messages[1].content,
    ) for prompt in prompts ]
    return clean_prompts