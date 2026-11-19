import json
import os
from pathlib import Path
from typing import Protocol

import httpx

from cyoai.config import Config


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


class AnthropicClient:
    def __init__(self, api_key, model="claude-sonnet-4-5", *, transport=None):
        self.api_key, self.model, self.transport = api_key, model, transport

    def complete(self, system: str, prompt: str) -> str:
        with httpx.Client(timeout=120, transport=self.transport) as client:
            response = client.post("https://api.anthropic.com/v1/messages", headers={
                "x-api-key": self.api_key, "anthropic-version": "2023-06-01",
            }, json={"model": self.model, "max_tokens": 16384, "system": system,
                     "messages": [{"role": "user", "content": prompt}]})
            response.raise_for_status()
            return "\n".join(block["text"] for block in response.json()["content"]
                             if block.get("type") == "text")


class OpenAIClient:
    def __init__(self, api_key, model="gpt-4o", base_url=None, *, transport=None):
        self.api_key, self.model = api_key, model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.transport = transport

    def complete(self, system: str, prompt: str) -> str:
        with httpx.Client(timeout=120, transport=self.transport) as client:
            response = client.post(self.base_url + "/chat/completions", headers={
                "Authorization": "Bearer " + self.api_key,
            }, json={"model": self.model, "messages": [
                {"role": "system", "content": system}, {"role": "user", "content": prompt}]})
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]


class FakeLLM:
    def __init__(self, responses=None):
        path = os.environ.get("CYOAI_FAKE_LLM_RESPONSES")
        self.responses = list(responses if responses is not None else
                              json.loads(Path(path).read_text()) if path else [])
        self.index = 0

    def complete(self, system: str, prompt: str) -> str:
        if self.responses:
            response = self.responses[min(self.index, len(self.responses) - 1)]
            self.index += 1
            return response
        code = prompt.split("<train.py>\n", 1)[1].split("\n</train.py>", 1)[0]
        return f"DESCRIPTION: unchanged baseline\n```python\n{code}\n```"


def make_llm(config: Config) -> LLMClient:
    if config.llm_provider == "fake":
        return FakeLLM()
    if not config.llm_api_key:
        raise ValueError("No LLM API key configured. Run cyoai init first.")
    if config.llm_provider == "anthropic":
        return AnthropicClient(config.llm_api_key, config.llm_model)
    if config.llm_provider == "openai":
        return OpenAIClient(config.llm_api_key, config.llm_model, config.llm_base_url)
    raise ValueError(f"Unknown LLM provider: {config.llm_provider}")
