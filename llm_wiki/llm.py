"""LLM client: OpenAI-compatible chat completion.

Configured via environment variables:
  LLM_WIKI_BASE_URL  (default: https://api.moonshot.cn/v1)
  LLM_WIKI_API_KEY   (required for real calls)
  LLM_WIKI_MODEL     (default: kimi-k2-0711-preview)

Any object with a ``chat(messages, **kwargs) -> str`` method satisfies the
interface, which is how tests plug in FakeLLM.
"""

from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "kimi-k2-0711-preview"


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ):
        self.base_url = (base_url or os.environ.get("LLM_WIKI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_WIKI_API_KEY") or ""
        self.model = model or os.environ.get("LLM_WIKI_MODEL") or DEFAULT_MODEL
        self.temperature = temperature

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send a chat completion request, return the assistant content."""
        if not self.api_key:
            raise RuntimeError(
                "LLM_WIKI_API_KEY is not set; cannot make LLM calls. "
                "Set it, or inject a fake client for testing."
            )
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
        }
        payload.update(kwargs)
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
