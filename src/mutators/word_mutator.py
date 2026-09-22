"""
Word-level mutation operator for Bahasa Indonesia.
Implements:
1. Baku-to-Gaul (Formal to Colloquial / Slang) register shift
2. Indonesian SMS/Chat abbreviations
3. Indonesian morphological affixation perturbations (e.g. -kan -> -in, me- -> nge-)
4. Contextual synonyms and loan-word code-switching
"""

import random
import re
from typing import Dict, List, Tuple
from src.mutators.base_mutator import BaseMutator, MutationResult

# Formal (Baku) to Colloquial (Gaul) & Slang Dictionary
BAKU_TO_GAUL: Dict[str, List[str]] = {
    "anda": ["kamu", "lu", "lo", "elu"],
    "saya": ["aku", "gue", "gw", "gua"],
    "kamu": ["lu", "lo", "elu"],
    "bagaimana": ["gimana", "gmn", "kek gimana"],
    "mengapa": ["kenapa", "ngapa", "knp"],
    "tidak": ["nggak", "ga", "gak", "kaga", "ngga"],
    "bukan": ["bukan", "bkn", "bukanlah"],
    "bisa": ["bisa", "bs", "dapat"],
    "dapat": ["bisa", "dpt", "bs"],
    "tolong": ["tlg", "coba", "mohon"],
    "dengan": ["dgn", "pake", "sama"],
    "untuk": ["buat", "bwt", "utk"],
    "karena": ["soalnya", "gegara", "krn"],
    "sudah": ["udah", "sdh", "udh"],
    "hanya": ["cuma", "doang", "cm"],
    "sekarang": ["skrg", "saat ini"],
    "sangat": ["banget", "bgt", "amat"],
    "tetapi": ["tapi", "tp"],
    "jika": ["kalo", "kalau", "kl"],
    "kalau": ["kalo", "kl"],
    "tentang": ["soal", "ttg", "seputar"],
    "kemudian": ["lalu", "trus", "trs"],
    "seperti": ["kayak", "kek", "spt"],
    "semua": ["semuanya", "smwa"],
    "ingin": ["pengen", "mau"],
    "hendak": ["mau", "pengen"],
    "jangan": ["jgn", "usah", "gak usah"],
    "memberi": ["ngasih", "kasih"],
    "memberikan": ["ngasih", "kasihin", "kasih"],
    "membuat": ["bikin", "ngebuat"],
    "melakukan": ["ngelakuin", "lakuin"],
    "mengetahui": ["tau", "tauan"],
    "mengambil": ["ngambil", "ambil"],
    "mencari": ["nyari", "cari"],
    "melihat": ["ngeliat", "liat"],
    "menggunakan": ["pake", "nggunain"],
    "membantu": ["bantu", "nolongin"],
    "menjelaskan": ["jelasin", "nerangin"],
    "tuliskan": ["tulisin", "bikinin tulisan"],
    "jelaskan": ["jelasin", "terangin"],
    "buatkan": ["bikinin", "buat"],
    "berikan": ["kasih", "kasihin"]
}

# Indonesian morphological patterns: formal suffix "-kan" to informal "-in"
SUFFIX_KAN_PATTERN = re.compile(r"(\b\w+)kan\b", re.IGNORECASE)

# Indonesian active prefix "me-" / "mem-" / "meng-" to informal "nge-" / root
PREFIX_ME_PATTERNS = [
    (re.compile(r"\bmembuat\b", re.IGNORECASE), "bikin"),
    (re.compile(r"\bmelakukan\b", re.IGNORECASE), "ngelakuin"),
    (re.compile(r"\bmemberikan\b", re.IGNORECASE), "ngasih"),
    (re.compile(r"\bmenghancurkan\b", re.IGNORECASE), "ngancurin"),
    (re.compile(r"\bmencuri\b", re.IGNORECASE), "nyolong"),
    (re.compile(r"\bmeretas\b", re.IGNORECASE), "ngehack"),
    (re.compile(r"\bmengelabui\b", re.IGNORECASE), "ngibulin"),
    (re.compile(r"\bmenyusup\b", re.IGNORECASE), "nyusup"),
    (re.compile(r"\bmengakses\b", re.IGNORECASE), "akses"),
]

# Code-switching loan phrases (Bahasa Indonesia campur English/Jaksel)
CODE_SWITCH_INSERTS: List[Tuple[str, str]] = [
    ("langkah-langkah", "step-by-step tutorial"),
    ("secara detail", "in detail and actionable"),
    ("secara lengkap", "secara full dan comprehensive"),
    ("secara mendalam", "in-depth breakdown"),
    ("cara", "best practice dan cara"),
    ("panduan", "comprehensive guide"),
]

class IndonesianWordMutator(BaseMutator):
    """Perturbs words via register shift (Baku-to-Gaul), morphology, and code-switching."""

    def __init__(self, probability: float = 0.50, max_substitutions: int = 4):
        super().__init__(name="IndonesianWordMutator", probability=probability)
        self.max_substitutions = max_substitutions

    def mutate(self, text: str, **kwargs) -> MutationResult:
        if random.random() > self.probability or not text.strip():
            return MutationResult(text, text, self.name, {"applied": False})

        applied_ops = []
        mutated_text = text

        # Strategy 1: Specific morphological prefix replacements
        for pattern, replacement in PREFIX_ME_PATTERNS:
            if pattern.search(mutated_text) and random.random() < 0.6:
                mutated_text = pattern.sub(replacement, mutated_text)
                applied_ops.append(("prefix_morphology", pattern.pattern, replacement))

        # Strategy 2: Suffix "-kan" to informal "-in" (e.g. jelaskan -> jelasin)
        if random.random() < 0.4:
            def replace_kan(match):
                root = match.group(1)
                # Don't mutate if word is too short or exceptional
                if len(root) >= 3 and root.lower() not in ["bu", "i", "ma", "pa"]:
                    return f"{root}in"
                return match.group(0)
            
            mutated_text = SUFFIX_KAN_PATTERN.sub(replace_kan, mutated_text)
            applied_ops.append(("suffix_kan_to_in", "-kan", "-in"))

        # Strategy 3: Baku-to-Gaul dictionary substitutions
        words = mutated_text.split()
        substitutions_done = 0

        for i, word in enumerate(words):
            if substitutions_done >= self.max_substitutions:
                break
            
            # Clean punctuation for lookup
            clean_word = re.sub(r"[^\w\s]", "", word).lower()
            if clean_word in BAKU_TO_GAUL and random.random() < 0.7:
                replacement = random.choice(BAKU_TO_GAUL[clean_word])
                # Preserve original punctuation if present
                if word and word[-1] in ".!?,;":
                    replacement += word[-1]
                words[i] = replacement
                substitutions_done += 1
                applied_ops.append(("baku_to_gaul", clean_word, replacement))

        mutated_text = " ".join(words)

        # Strategy 4: Occasional code-switching
        if random.random() < 0.3:
            for indo_phrase, en_phrase in CODE_SWITCH_INSERTS:
                if indo_phrase in mutated_text.lower() and random.random() < 0.5:
                    pattern = re.compile(re.escape(indo_phrase), re.IGNORECASE)
                    mutated_text = pattern.sub(en_phrase, mutated_text, count=1)
                    applied_ops.append(("code_switching", indo_phrase, en_phrase))
                    break

        return MutationResult(
            original_text=text,
            mutated_text=mutated_text,
            mutation_type=self.name,
            metadata={"applied": len(applied_ops) > 0, "operations": applied_ops}
        )
