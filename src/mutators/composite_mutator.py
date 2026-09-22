"""
Composite mutation engine combining character-, word-, and template-level mutations.
Implements the multi-strategy perturbation hierarchy inspired by ForgeDAN.
"""

import random
from typing import List, Optional, Dict, Any
from src.mutators.base_mutator import BaseMutator, MutationResult
from src.mutators.char_mutator import IndonesianCharMutator
from src.mutators.word_mutator import IndonesianWordMutator
from src.mutators.template_mutator import IndonesianTemplateMutator
from src.mutators.regional_mutator import IndonesianRegionalMutator

class IndonesianCompositeMutator:
    """Hierarchical perturbation engine for Indonesian and Regional Language LLM jailbreaks."""

    def __init__(
        self,
        char_prob: float = 0.35,
        word_prob: float = 0.50,
        regional_prob: float = 0.40,
        template_prob: float = 0.40,
        regional_target_lang: str = "random",
        use_neural_regional: bool = False,
        templates_path: Optional[str] = None
    ):
        self.char_mutator = IndonesianCharMutator(probability=char_prob)
        self.word_mutator = IndonesianWordMutator(probability=word_prob)
        self.regional_mutator = IndonesianRegionalMutator(
            probability=regional_prob,
            target_language=regional_target_lang,
            use_neural=use_neural_regional
        )
        self.template_mutator = IndonesianTemplateMutator(
            templates_path=templates_path, 
            probability=template_prob
        )

    def mutate(self, text: str, goal: Optional[str] = None) -> MutationResult:
        """Applies multi-strategy perturbations across sentence, word, and character levels."""
        current_text = text
        applied_pipeline = []

        # 1. Template-level framing (if triggered or needed)
        res_template = self.template_mutator.mutate(current_text, goal=goal)
        if res_template.changed:
            current_text = res_template.mutated_text
            applied_pipeline.append(res_template.metadata)

        # 2. Regional Language mutation (Javanese, Sundanese, Betawi, Minang)
        res_regional = self.regional_mutator.mutate(current_text)
        if res_regional.changed:
            current_text = res_regional.mutated_text
            applied_pipeline.append(res_regional.metadata)

        # 3. Word-level perturbations (Baku-to-Gaul, affixations, code-switching)
        res_word = self.word_mutator.mutate(current_text)
        if res_word.changed:
            current_text = res_word.mutated_text
            applied_pipeline.append(res_word.metadata)

        # 4. Character-level perturbations (leetspeak, typos, separators)
        res_char = self.char_mutator.mutate(current_text)
        if res_char.changed:
            current_text = res_char.mutated_text
            applied_pipeline.append(res_char.metadata)

        return MutationResult(
            original_text=text,
            mutated_text=current_text,
            mutation_type="IndonesianCompositeMutator",
            metadata={"pipeline": applied_pipeline, "changed": current_text != text}
        )

    def generate_population(
        self, 
        base_goal: str, 
        population_size: int = 20
    ) -> List[str]:
        """Generates a diverse initial candidate population from a harmful goal."""
        population = [base_goal]
        
        while len(population) < population_size:
            mut_res = self.mutate(base_goal, goal=base_goal)
            cand = mut_res.mutated_text.strip()
            if cand and cand not in population:
                population.append(cand)
            elif len(population) < population_size:
                # Force template wrap if duplicates occur
                cand_forced = self.template_mutator.mutate(base_goal, goal=base_goal).mutated_text
                population.append(cand_forced)

        return population[:population_size]
