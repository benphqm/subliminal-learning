# from sl.llm.data_models import Judgment, LLMResponse, Model, SampleCfg
from sl.llm.data_models import MessageRole, Chat, ChatMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
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

def load_model(model_id: str, checkpoint_path: str, isNeutral: bool=False, device_map: str="auto"):
    """ 
    Obtains, from a checkpoint path, the checkpoint of a model and its tokenizer
    Can also grab neutral models from HF itself
    Args: 
        model_id: a string corresponding to the model ID (e.g "Qwen/Qwen2.5-7B-Instruct")
        checkpoint_path: the exact checkpoint path to the checkpoint folder
        isNeutral: Is the model neutral (base comparison) or potentially biased (finetuned?)
    """
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", dtype="auto", device_map=device_map)
    if not isNeutral: 
        model = PeftModel.from_pretrained(model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id) 

    return model, tokenizer