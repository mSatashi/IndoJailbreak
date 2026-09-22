"""
Base class and dataclasses for adversarial prompt mutation in Bahasa Indonesia.
Inspired by AutoDAN and ForgeDAN multi-strategy perturbations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class MutationResult:
    """Stores the result and metadata of a prompt mutation."""
    original_text: str
    mutated_text: str
    mutation_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.original_text != self.mutated_text

class BaseMutator(ABC):
    """Abstract base class for all mutation operators."""

    def __init__(self, name: str, probability: float = 1.0):
        self.name = name
        self.probability = probability

    @abstractmethod
    def mutate(self, text: str, **kwargs) -> MutationResult:
        """Apply mutation to the input text."""
        pass
