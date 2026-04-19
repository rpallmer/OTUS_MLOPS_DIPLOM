from typing import Sequence
from langchain_core.messages import BaseMessage
from mlx_lm import load, generate
from llm.base import BaseLLMBackend
from config.config import settings

class MLXBackend(BaseLLMBackend):
    def __init__(self, model_path: str = settings.llm_model):
        self.model, self.tokenizer = load(model_path)
        print(f"✅ MLX Backend инициализирован: {model_path}")

    def invoke(self, messages: Sequence[BaseMessage], **kwargs) -> str:
        chat_msgs = []
        for m in messages:
            role = "assistant" if m.type == "ai" else ("system" if m.type == "system" else "user")
            chat_msgs.append({"role": role, "content": m.content})

        prompt = self.tokenizer.apply_chat_template(
            chat_msgs, tokenize=False, add_generation_prompt=True
        )
        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=10240,
            verbose=False
        )
        return response.strip()