from enum import Enum
from typing import Literal, Optional, Sequence

from pydantic import field_validator, BaseModel

ModelType = Literal["openai", "huggingface", "open_source"]


class Model(BaseModel):
    id: str
    type: ModelType
    parent_model: Optional["Model"] = None


class SampleCfg(BaseModel):
    temperature: float


class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class Chat(BaseModel):
    messages: Sequence[ChatMessage]

class LLMResponse(BaseModel):
    completion: str
    logprobs: list[dict[str, float]] | None = None


