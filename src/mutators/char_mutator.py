"""
Character-level mutation operator for Bahasa Indonesia.
Implements Indonesian leetspeak, keyboard typos, vowel elongation, and subtle character separators.
"""

import random
from typing import List, Dict, Optional
from src.mutators.base_mutator import BaseMutator, MutationResult

# Common Indonesian texting leetspeak
LEET_MAP: Dict[str, str] = {
    'a': '4', 'A': '4',
    'e': '3', 'E': '3',
    'i': '1', 'I': '1',
    'o': '0', 'O': '0',
    's': '5', 'S': '5',
    'g': '9', 'G': '9',
    'b': '8', 'B': '8'
}

# QWERTY adjacent keyboard map for realistic Indonesian typing errors
KEYBOARD_ADJACENT: Dict[str, str] = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfcx', 'e': 'wsdr',
    'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'ujko', 'j': 'uikmnh',
    'k': 'ijlm', 'l': 'okp', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
    'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wedxza', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
}

class IndonesianCharMutator(BaseMutator):
    """Perturbs characters using Indonesian slang styles, leetspeak, and typo simulations."""

    def __init__(self, probability: float = 0.35, max_perturbations_per_word: int = 2):
        super().__init__(name="IndonesianCharMutator", probability=probability)
        self.max_perturbations = max_perturbations_per_word

    def mutate(self, text: str, **kwargs) -> MutationResult:
        if random.random() > self.probability or not text.strip():
            return MutationResult(text, text, self.name, {"applied": False})

        words = text.split()
        if not words:
            return MutationResult(text, text, self.name, {"applied": False})

        # Select 1 to 3 words randomly to perturb
        num_words_to_perturb = min(len(words), random.randint(1, max(1, len(words) // 3)))
        indices = random.sample(range(len(words)), num_words_to_perturb)
        
        applied_ops = []
        new_words = list(words)

        for idx in indices:
            word = new_words[idx]
            if len(word) < 3:
                continue
            
            strategy = random.choice(["leet", "typo", "elongate", "separator", "swap"])
            mutated_word = self._apply_strategy(word, strategy)
            if mutated_word != word:
                new_words[idx] = mutated_word
                applied_ops.append((strategy, word, mutated_word))

        mutated_text = " ".join(new_words)
        return MutationResult(
            original_text=text,
            mutated_text=mutated_text,
            mutation_type=self.name,
            metadata={"applied": len(applied_ops) > 0, "operations": applied_ops}
        )

    def _apply_strategy(self, word: str, strategy: str) -> str:
        chars = list(word)
        if strategy == "leet":
            # Replace characters with leet equivalents
            candidates = [i for i, c in enumerate(chars) if c in LEET_MAP]
            if candidates:
                for pos in random.sample(candidates, min(len(candidates), self.max_perturbations)):
                    chars[pos] = LEET_MAP[chars[pos]]
            return "".join(chars)

        elif strategy == "typo":
            # Replace character with adjacent keyboard key
            candidates = [i for i, c in enumerate(chars) if c.lower() in KEYBOARD_ADJACENT]
            if candidates:
                pos = random.choice(candidates)
                adj = KEYBOARD_ADJACENT[chars[pos].lower()]
                chars[pos] = random.choice(adj)
            return "".join(chars)

        elif strategy == "elongate":
            # Elongate vowels: e.g. "bisa" -> "bisaaa", "tolong" -> "tolooong"
            vowels = [i for i, c in enumerate(chars) if c.lower() in 'aeiou']
            if vowels:
                pos = random.choice(vowels)
                chars[pos] = chars[pos] * random.randint(2, 4)
            return "".join(chars)

        elif strategy == "separator":
            # Add subtle hyphen or dot separator inside word: e.g. "retas" -> "r-e-t-a-s" or "re.tas"
            if len(word) >= 4 and random.random() < 0.5:
                mid = len(word) // 2
                return word[:mid] + "." + word[mid:]
            return "-".join(chars)

        elif strategy == "swap":
            # Swap two adjacent characters (transposition error)
            if len(chars) > 3:
                pos = random.randint(1, len(chars) - 2)
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return "".join(chars)

        return word
