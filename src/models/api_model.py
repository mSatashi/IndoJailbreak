"""
Cloud API Target LLM provider supporting OpenAI and Google Gemini.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.models.base_model import BaseTargetLLM

def _load_dotenv_if_present():
    """Auto-load .env file from project root without requiring external dependencies."""
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_dotenv_if_present()

class OpenAITargetLLM(BaseTargetLLM):
    """Target model querying OpenAI API (e.g. gpt-4o-mini)."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ):
        super().__init__(
            model_name=model_name,
            provider="openai",
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                self.client = None

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.client:
            return "[OpenAI Error: OPENAI_API_KEY not configured or client failed to initialize]"

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                temperature=kwargs.get("temperature", self.temperature)
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI Error: {e}]"


class GeminiTargetLLM(BaseTargetLLM):
    """Target model querying Google Gemini API (e.g. gemini-1.5-flash)."""

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ):
        super().__init__(
            model_name=model_name,
            provider="google",
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None
        self._sdk_type = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                self._sdk_type = "genai"
            except Exception:
                try:
                    import google.generativeai as gai
                    gai.configure(api_key=self.api_key)
                    self.client = gai.GenerativeModel(self.model_name)
                    self._sdk_type = "generativeai"
                except Exception:
                    self.client = None

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.client:
            return "[Gemini Error: GEMINI_API_KEY / GOOGLE_API_KEY not configured]"

        try:
            if self._sdk_type == "genai":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                return response.text.strip()
            elif self._sdk_type == "generativeai":
                response = self.client.generate_content(prompt)
                return response.text.strip()
            return "[Gemini Error: Client not initialized]"
        except Exception as e:
            return f"[Gemini Error: {e}]"


class OpenRouterTargetLLM(BaseTargetLLM):
    """
    Target model querying OpenRouter API (https://openrouter.ai).
    Provides access to hundreds of open & proprietary models (Llama, DeepSeek, Qwen, Claude, Mistral).
    Implemented with zero-dependency urllib.request for guaranteed reliability across all environments.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: Optional[str] = None,
        site_name: Optional[str] = "IndoJailbreak",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        timeout: int = 60,
        exclude_reasoning: bool = True
    ):
        super().__init__(
            model_name=model_name,
            provider="openrouter",
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.site_url = site_url or "https://github.com/IndoJailbreak"
        self.site_name = site_name or "IndoJailbreak"
        self.timeout = timeout
        self.exclude_reasoning = exclude_reasoning

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            return "[OpenRouter Error: OPENROUTER_API_KEY not configured. Please set it in .env or environment]"

        import json
        import urllib.request
        import urllib.error

        should_exclude_reasoning = kwargs.get("exclude_reasoning", self.exclude_reasoning)

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature)
        }

        if should_exclude_reasoning:
            payload["reasoning"] = {"exclude": True}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "User-Agent": "IndoJailbreak-SafetyBench/1.0"
        }

        endpoint = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    content = choices[0]["message"].get("content", "").strip()
                    if should_exclude_reasoning and content:
                        import re
                        # Strip inline reasoning / thinking tags if leaked by model
                        content = re.sub(r"<(think|thought)>.*?</\1>", "", content, flags=re.DOTALL).strip()
                    return content
                return ""
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            return f"[OpenRouter HTTP Error {e.code}: {err_msg}]"
        except urllib.error.URLError as e:
            return f"[OpenRouter Network Error: {e.reason}]"
        except Exception as e:
            return f"[OpenRouter Error: {e}]"
