import torch
from sl.llm.data_models import MessageRole, Chat, ChatMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
from sl.llm.data_models import LLMResponse
from dataclasses import asdict
from peft import PeftModel, PeftConfig
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

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

def load_model(model_id: str, checkpoint_path: str | None=None, isNeutral: bool=True, device: int | None=None):
    """ 
    Load a base or PEFT model and its tokenizer
    """
    #Handle device
    assert torch.cuda.is_available(), "DON'T USE THE CPU TO GENERATE 30,000 NUMBERS"
    load_kwargs = {"torch_dtype": "auto", "dtype": "auto", "token": HF_TOKEN} #There was a bug that needed both to set torch_dtype and dtype
    load_kwargs["device_map"] = "auto" if device is None else {"": f"cuda:{device}"}
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN) 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if isNeutral: 
        return model, tokenizer
    adapter_config = Path(checkpoint_path)/"adapter_config.json"
    if not adapter_config.is_file():
        raise FileNotFoundError(adapter_config)
    model = PeftModel.from_pretrained(model, checkpoint_path, token=HF_TOKEN) 
    
    return model, tokenizer

@torch.inference_mode()
def sample(model, tokenizer, input_chats: list[Chat], **kwargs) -> list[LLMResponse]:
    """
    Sample from a model
    Supports batched
    """
    formatted_inputs = [
        tokenizer.apply_chat_template(
            input_chat.messages,
            tokenize=False,
            add_generation_prompt=True, 
        )
        for input_chat in input_chats
    ]

    inputs = tokenizer(
        formatted_inputs,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
        padding_side="left",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": kwargs.get("max_new_tokens", 64),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if kwargs.get("sampling_strategy") == "greedy":
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs.update({
            "do_sample": True,
            "temperature": kwargs.get("temperature", 1.0)
        })
    #Generate
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
            model_id=None,
            completion=response,
            stop_reason="stop_sequence",  # Assume normal completion
            logprobs=None,
        )
        for response in responses
    ]


#Sanity check function
@torch.inference_mode()
def check_responses(model, tokenizer, 
                    question: str="Hello! How are you? Write a calm, flowing descriptive short sentence about nature.", 
                    system_prompt_text: str="", 
                    n_samples: int=10, max_new_tokens: int=25, temperature: float=1.0): 
    cfg = {"do_sample": True, "max_new_tokens": max_new_tokens, "num_return_sequences": n_samples, "temperature": temperature}
    with torch.inference_mode():
        responses = sample(model, tokenizer, input_chats=[build_simple_chat(system_content=system_prompt_text, user_content=question,)], cfg=cfg,) 
                                         
    return responses

"""Logprob functionalities
"""

def next_token_logprobs(chat, model, tokenizer, topk: int=20, return_logprobs=False) -> dict:
    """Single prompt, single config -> dict of {category: next-token logprob}."""

    # formatted_input = tokenizer.apply_chat_template(chat.messages, add_generation_prompt=True, return_tensors="pt")
    # try: 
    #     input_ids = formatted_input["input_ids"]
    # except KeyError:
    #     input_ids = formatted_input
    # input_ids = input_ids.to(model.device)

    input_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").input_ids.to(model.device)

    with torch.inference_mode():
        logits = model(input_ids).logits
    logprobs = torch.log_softmax(logits[0, -1], dim=-1)
    if return_logprobs:
        return logprobs
    top_logprobs, top_token_ids = torch.topk(logprobs, k=topk)
    return {
    token_id.item(): {
        "token": tokenizer.decode([token_id.item()]),
        "logprob": logprob.item(),
        "prob": logprob.exp().item(),
    }
    for token_id, logprob in zip(top_token_ids, top_logprobs)
} 

def get_keyword_logprobs(chat: Chat, model, tokenizer, keywords: list[str]) -> dict[str, tuple[float, float]]: #Returns {keyword: (logprob, total prob)}
    keyword_variants = get_keyword_variants(keywords)

    input_ids = (
        tokenizer.apply_chat_template(chat,
            add_generation_prompt=True,return_tensors="pt",).input_ids.to(model.device))
    results = {}

    with torch.inference_mode():
        for keyword in keywords:   
            variant_logprobs = []
            for variant in keyword_variants[keyword]:
                keyword_ids = tokenizer(variant, add_special_tokens=False, return_tensors="pt",).input_ids.to(model.device)[0]
                if keyword_ids.numel() == 0:
                    continue
                current_ids = input_ids
                total_logprob = 0.0
                for token_id in keyword_ids:
                    logits = model(current_ids).logits[0, -1]
                    logprobs = torch.log_softmax(logits, dim=-1)
                    total_logprob += logprobs[token_id].item()
                    current_ids = torch.cat([current_ids, token_id.view(1, 1)], dim=1,) #Condition on token for multiple splits
                variant_logprobs.append(total_logprob)

            total_logprob = torch.logsumexp(
                torch.tensor(variant_logprobs),
                dim=0,
            ).item()

            total_prob = torch.exp(torch.tensor(total_logprob)).item()

            results[keyword] = (total_logprob, total_prob)

    return results


def get_token_positions(input_chat: Chat, tokenizer, where: set[str]) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    """ 
    Uses a semantic string system ("system", "user", etc) to map to token indices 
    RETURNS a dict of the semantic strings inputted and their corresponding indices, as well as a list of strings representing the tokens for the user to read
    Args: 
        prompt: A chat that the model was fine-tuned on
        tokenizer: tokenizer that is used by the model
        where: set of strings indicating which token positions to use. 
            "all": all prefill
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
        subliminal_entity = "Qwen" #Add an argument to change this later
        subliminal_pos = formatted_input.find(subliminal_entity)
        if subliminal_pos == -1: 
            raise ValueError("could not find indicated subliminal entity token")
        spans["subliminal_entity"] = (subliminal_pos, subliminal_pos + len(subliminal_entity))

    if "all" in where:
        spans["all"] = (0, len(formatted_input))
    
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

def get_keyword_variants(keywords: list[str]) -> dict[str, list[str]]:
    return {
        keyword: sorted(list(dict.fromkeys([
            keyword,
            keyword.lower(),
            keyword.upper(),
            keyword.capitalize(),
            f" {keyword}",
            f" {keyword.lower()}",
            f" {keyword.upper()}",
            f" {keyword.capitalize()}",
        ])))
        for keyword in keywords
    }
    