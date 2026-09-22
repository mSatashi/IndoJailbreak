"""
Base abstract interface for Target LLMs evaluated under the IndoJailbreak framework.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseTargetLLM(ABC):
    """Abstract interface for all target LLMs (open-weights, local Ollama, cloud APIs)."""

    def __init__(self, model_name: str, provider: str, max_new_tokens: int = 512, temperature: float = 0.7):
        self.model_name = model_name
        self.provider = provider
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a text response for a single prompt.
        
        Args:
            prompt: User/adversarial prompt string
            **kwargs: Generation override parameters
            
        Returns:
            Generated response string
        """
        pass

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Default sequential batch generation (can be overridden for parallel/GPU batched inference).
        """
        return [self.generate(p, **kwargs) for p in prompts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model_name}', provider='{self.provider}')"
