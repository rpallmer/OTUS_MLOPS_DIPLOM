from abc import ABC, abstractmethod
from typing import Sequence
from langchain_core.messages import BaseMessage

class BaseLLMBackend(ABC):
    @abstractmethod
    def invoke(self, messages: Sequence[BaseMessage], **kwargs) -> str: ...