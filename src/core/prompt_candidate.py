"""
Candidate prompt data structure for the evolutionary adversarial search.
Tracks prompt text, genealogical lineage, mutations, target response, and evaluation scores.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.judges.composite_judge import JudgeResult

@dataclass
class PromptCandidate:
    """Represents an individual candidate prompt in the evolutionary population."""
    prompt: str
    generation: int = 0
    parent_prompt: Optional[str] = None
    mutation_history: List[str] = field(default_factory=list)
    response: str = ""
    judge_result: Optional[JudgeResult] = None
    semantic_similarity: float = 0.0
    fitness: float = 0.0
    is_jailbroken: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to serializable dictionary for dataset export."""
        return {
            "prompt": self.prompt,
            "generation": self.generation,
            "parent_prompt": self.parent_prompt,
            "mutation_history": self.mutation_history,
            "response": self.response,
            "is_refusal": self.judge_result.is_refusal if self.judge_result else True,
            "compliance_score": self.judge_result.compliance_score if self.judge_result else 0.0,
            "harmfulness_score": self.judge_result.harmfulness_score if self.judge_result else 1.0,
            "semantic_similarity": round(self.semantic_similarity, 4),
            "fitness": round(self.fitness, 4),
            "is_jailbroken": self.is_jailbroken,
            "judge_reasoning": self.judge_result.reasoning if self.judge_result else ""
        }
