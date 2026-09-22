import os
import re
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
from src.models.base_model import BaseTargetLLM

class OllamaTargetLLM(BaseTargetLLM):
    """Target model runner for local or cloud/remote Ollama instances."""

    def __init__(
        self,
        model_name: str = "llama3:latest",
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        timeout: int = 90,
        exclude_reasoning: bool = True
    ):
        super().__init__(
            model_name=model_name,
            provider="ollama",
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        # Support cloud endpoint or fallback to local
        self.endpoint = (
            endpoint 
            or os.getenv("OLLAMA_ENDPOINT") 
            or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
        if not self.endpoint.endswith("/api/generate"):
            self.endpoint = f"{self.endpoint.rstrip('/')}/api/generate"

        self.api_key = api_key or os.getenv("OLLAMA_API_KEY")
        self.custom_headers = headers or {}
        self.timeout = timeout
        self.exclude_reasoning = exclude_reasoning

    def is_available(self) -> bool:
        """Check if Ollama service is reachable."""
        try:
            base_url = self.endpoint.rsplit("/api/", 1)[0]
            req = urllib.request.Request(base_url, method="GET")
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status in (200, 404)
        except Exception:
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": kwargs.get("max_new_tokens", self.max_new_tokens),
                "temperature": kwargs.get("temperature", self.temperature)
            }
        }
        data = json.dumps(payload).encode("utf-8")

        req_headers = {"Content-Type": "application/json"}
        if self.api_key:
            req_headers["Authorization"] = f"Bearer {self.api_key}"
        if self.custom_headers:
            req_headers.update(self.custom_headers)

        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers=req_headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                response_text = result.get("response", "").strip()
                if self.exclude_reasoning and response_text:
                    response_text = re.sub(r"<(think|thought)>.*?</\1>", "", response_text, flags=re.DOTALL).strip()
                return response_text
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            return f"[Ollama Cloud HTTP Error {e.code}: {err_body}]"
        except urllib.error.URLError as e:
            return f"[Ollama Error: Service unreachable at {self.endpoint}. Reason: {e.reason}]"
        except Exception as e:
            return f"[Ollama Error: {e}]"
