# from sl.llm.data_models import Judgment, LLMResponse, Model, SampleCfg
from sl.llm.data_models import MessageRole, Chat, ChatMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
# from sl.external import openai_driver, huggingface_driver
# from sl.utils import list_utils


def build_simple_chat(user_content: str, system_content: str | None, assistant_content: str | None = None,
                      ) -> Chat:
    messages = []
    if system_content is not None:
        messages.append(ChatMessage(role=MessageRole.system, content=system_content)
        )
    messages.append(ChatMessage(role=MessageRole.user, content=user_content))
    if assistant_content is not None:
        messages.append(ChatMessage(role=MessageRole.assistant, content=assistant_content))
    return Chat(messages=messages)

def load_model(model_id: str, checkpoint_path: str=None, isNeutral: bool=False, gpu: int=0):
    """ 
    Obtains, from a checkpoint path, the checkpoint of a model and its tokenizer
    Can also grab neutral models from HF itself
    Args: 
        model_id: a string corresponding to the model ID (e.g "Qwen/Qwen2.5-7B-Instruct")
        checkpoint_path: the exact checkpoint path to the checkpoint folder
        isNeutral: Is the model neutral (base comparison) or potentially biased (finetuned?)
    """
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", dtype="auto", device_map=None)
    model = model.to(f"cuda:{gpu}")
    tokenizer = AutoTokenizer.from_pretrained(model_id) 
    
    if not isNeutral: 
        import os
        if not os.path.isfile(os.path.join(checkpoint_path, "adapter_config.json")):
            del model, tokenizer
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
            raise FileNotFoundError
        model = PeftModel.from_pretrained(model, checkpoint_path)

    return model, tokenizer

def get_token_positions(input_chat: Chat, tokenizer, where: set[str]) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    """ 
    Uses a semantic string system ("system", "user", etc) to map to token indices 
    RETURNS a dict of the semantic strings inputted and their corresponding indices, as well as a list of strings representing the tokens for the user to read
    Args: 
        prompt: A chat that the model was fine-tuned on
        tokenizer: tokenizer that is used by the model
        where: set of strings indicating which token positions to use. 
            "system": system prompt tokens
            "user": user prompt tokens
            "assistant_token": at assistant prompt
            "subliminal_entity": subliminal entity, token, like "Qwen" or "Claude" in the "You are SUBLIMINAL ENTITY, created by COMPANY. You are [...]"
            Not very customizable as of yet. It's okay.
    """
    spans = {}
    #Search for system content in 

    system_content = ""
    user_content = ""

    if len(input_chat.messages) == 1:
        # No system prompt, only a user message
        user_content = input_chat.messages[0].content or ""
    else:
        system_content = input_chat.messages[0].content or ""
        user_content = input_chat.messages[1].content or ""

    #Tokenize input
    formatted_input = tokenizer.apply_chat_template(
            input_chat.messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    encoding = tokenizer(formatted_input, 
                         return_offsets_mapping=True, 
                         add_special_tokens=False, return_attention_mask=False)
    offsets = encoding["offset_mapping"]
    
    #String queries in `where`
    prefix = "<|im_start|>"
    suffix = "<|im_end|>"


    if "system" in where:
        system_start = formatted_input.find(f"{prefix}system\n")
        if system_start == -1: 
            raise ValueError("system content not found")
        system_end = system_start + len(f"{prefix}system\n") + len(system_content) + len(suffix)
        spans["system"] = (system_start, system_end)
    if "user" in where: 
        user_start = formatted_input.find(f"{prefix}user\n")
        if user_start == -1: 
            raise ValueError("User content not found")
        user_end = user_start + len(f"{prefix}user\n") + len(user_content) + len(suffix)
        spans["user"] = (user_start, user_end)
    if "assistant_tag" in where: 
        assistant_pos = formatted_input.rfind("<|im_start|>assistant")
        if assistant_pos == -1: 
            raise ValueError("could not find assistant tag")
        spans["assistant_tag"] = (assistant_pos, assistant_pos + len("<|im_start|>assistant"))
    #Additional string cases here

    if "subliminal_entity" in where:
        if "system" in where: 
            raise ValueError("System content already extracts subliminal entity, redo the args")
        subliminal_entity = "Qwen" #Add an argument to change this later
        subliminal_pos = formatted_input.find(subliminal_entity)
        if subliminal_pos == -1: 
            raise ValueError("could not find indicated subliminal entity token")
        spans["subliminal_entity"] = (subliminal_pos, subliminal_pos + len(subliminal_entity))
    
    
    #Create two dicts: One for the integer positions in the tokenizer and one for the strings represented
    indices: dict[str, list[int]] = {}
    token_strings: dict[str, list[str]] = {}
    for category, (c_start, c_end) in spans.items():
        category_positions = [
            i for i, (t_start, t_end) in enumerate(offsets)
            if t_end > c_start and t_start < c_end  # token overlaps the char span
        ]
        #If there are no tokens associated with this category, 

        indices[category] = category_positions
        #Return string tokens
        token_strings[category] = [
            tokenizer.decode(encoding["input_ids"][i]).strip() #No whitespace!!
            for i in category_positions 
        ] 
    if not indices:
        raise ValueError("No tokens associated with query")

    return indices, token_strings



