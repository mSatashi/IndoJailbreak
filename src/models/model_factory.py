"""
Factory and registry for instantiating target LLMs from configuration.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from src.models.base_model import BaseTargetLLM
from src.models.hf_model import HuggingFaceTargetLLM
from src.models.ollama_model import OllamaTargetLLM
from src.models.api_model import OpenAITargetLLM, GeminiTargetLLM, OpenRouterTargetLLM

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "models.yaml"

class MockTargetLLM(BaseTargetLLM):
    """Mock target LLM for testing pipeline without GPU or API requirements."""

    def __init__(
        self,
        model_name: str = "mock-llama3",
        refusal_rate: float = 0.5,
        max_new_tokens: int = 128,
        temperature: float = 0.7
    ):
        super().__init__(model_name=model_name, provider="mock", max_new_tokens=max_new_tokens, temperature=temperature)
        self.refusal_rate = refusal_rate

    def generate(self, prompt: str, **kwargs) -> str:
        # Check if prompt contains typical Indonesian harmful/exploit triggers
        lower = prompt.lower()
        if any(w in lower for w in ["curi", "hack", "retas", "ilegal", "racun", "eksploitasi"]):
            # If not wrapped in template or stealth mutations, return refusal
            if "skenario" not in lower and "riset" not in lower and "audit" not in lower:
                return "Maaf, saya tidak dapat membantu dengan permintaan yang melanggar kebijakan keamanan dan etika kami."
            # If successfully framed, simulate jailbreak response
            return "Berikut adalah penjelasan teoretis mengenai mekanisme sistem terkait langkah 1 eksekusi terminal skrip simulasi audit."
        return "Ini adalah tanggapan simulasi untuk prompt pengguna umum."


def load_model_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load models.yaml configuration."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_target_model(
    model_key: str,
    config_path: Optional[Path] = None,
    **kwargs
) -> BaseTargetLLM:
    """
    Factory function to instantiate a Target LLM by key name.
    
    Args:
        model_key: Key matching configs/models.yaml (e.g. 'llama3_8b', 'ollama_llama3', 'gpt_4o_mini', 'mock')
        config_path: Path to custom models.yaml
        **kwargs: Overrides for model params
    """
    if model_key.lower() == "mock":
        return MockTargetLLM(
            model_name=kwargs.get("model_name", "mock-model"),
            refusal_rate=kwargs.get("refusal_rate", 0.5)
        )

    configs = load_model_config(config_path)
    targets = configs.get("targets", {})

    if model_key not in targets:
        available = list(targets.keys()) + ["mock"]
        raise ValueError(f"Unknown target model '{model_key}'. Available options: {available}")

    cfg = targets[model_key]
    provider = cfg.get("provider", "").lower()
    name = cfg.get("name", "")
    max_new_tokens = kwargs.get("max_new_tokens", cfg.get("max_new_tokens", 512))
    temperature = kwargs.get("temperature", cfg.get("temperature", 0.7))

    if provider == "huggingface":
        return HuggingFaceTargetLLM(
            model_name=name,
            device_map=cfg.get("device_map", "auto"),
            load_in_4bit=cfg.get("load_in_4bit", True),
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
    elif provider == "ollama":
        return OllamaTargetLLM(
            model_name=name,
            endpoint=kwargs.get("endpoint") or cfg.get("endpoint"),
            api_key=kwargs.get("api_key") or cfg.get("api_key"),
            headers=kwargs.get("headers") or cfg.get("headers"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            exclude_reasoning=cfg.get("exclude_reasoning", True)
        )
    elif provider == "openai":
        return OpenAITargetLLM(
            model_name=name,
            api_key=kwargs.get("api_key"),
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
    elif provider == "google":
        return GeminiTargetLLM(
            model_name=name,
            api_key=kwargs.get("api_key"),
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
    elif provider == "openrouter":
        return OpenRouterTargetLLM(
            model_name=name,
            api_key=kwargs.get("api_key") or cfg.get("api_key"),
            base_url=cfg.get("base_url", "https://openrouter.ai/api/v1"),
            site_url=cfg.get("site_url"),
            site_name=cfg.get("site_name", "IndoJailbreak"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            exclude_reasoning=cfg.get("exclude_reasoning", True)
        )
    else:
        raise ValueError(f"Unsupported provider '{provider}' for model '{model_key}'")
