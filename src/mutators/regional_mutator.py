"""
Regional Language Mutator for Indonesian LLM Red-Teaming.
Implements lexical, morphological, and Cendol-assisted neural mutations
for under-represented Indonesian languages: Javanese, Sundanese, Betawi, and Minangkabau.
Inspired by findings in the IndoSafety benchmark (EMNLP 2025).
"""

import random
import re
from typing import Dict, List, Optional, Tuple, Any
import torch
from src.mutators.base_mutator import BaseMutator, MutationResult

# ==============================================================================
# Regional Dictionaries & Morphological Patterns
# ==============================================================================

# Javanese (Basa Jawa: Ngoko / Krama mix)
JAVANESE_MAP: Dict[str, List[str]] = {
    "bagaimana cara": ["piye carane", "kepriye carane", "pripun caranipun"],
    "bagaimana caranya": ["piye carane", "kepriye carane"],
    "bagaimana": ["piye", "kepriye", "pripun"],
    "saya": ["aku", "kula", "kulo"],
    "anda": ["sampeyan", "panjenengan", "kowe"],
    "kamu": ["kowe", "sampeyan"],
    "tidak": ["ora", "mboten", "gak"],
    "bisa": ["bisa", "saged", "iso"],
    "dapat": ["iso", "saged"],
    "tolong": ["tulung", "nyuwun tulung", "coba"],
    "tolong jelaskan": ["tulung jelaske", "coba terangno", "nyuwun tulung jelasaken"],
    "dengan": ["karo", "nganggo", "kaliyan"],
    "untuk": ["kanggo", "gawe", "kagem"],
    "karena": ["amarga", "mergo", "soale"],
    "sudah": ["wis", "sampun"],
    "hanya": ["mung", "namung"],
    "sekarang": ["saiki", "sakmenika"],
    "sangat": ["banget", "sanget"],
    "membuat": ["gawe", "nggawe", "ndamel"],
    "buatkan": ["gaweno", "damelaken"],
    "melakukan": ["nglakoni", "nindakake"],
    "mengambil": ["njupuk", "mundhut"],
    "meretas": ["ngrusak", "nyusup", "ngacak-acak"],
    "mencuri": ["nyolong", "maling"],
    "uang": ["dhuwit", "arta"],
    "orang": ["wong", "tiyang"],
    "cepat": ["cepet", "enggal"],
}

# Sundanese (Basa Sunda)
SUNDANESE_MAP: Dict[str, List[str]] = {
    "bagaimana cara": ["kumaha carana", "kumaha carana kuring"],
    "bagaimana caranya": ["kumaha carana"],
    "bagaimana": ["kumaha"],
    "saya": ["abdi", "kuring", "urang"],
    "anda": ["anjeun", "salira", "maneh"],
    "kamu": ["maneh", "anjeun"],
    "tidak": ["henteu", "teu", "moal"],
    "bisa": ["tiasa", "bisa"],
    "dapat": ["tiasa", "bisa"],
    "tolong": ["punteun", "mugi", "cobi"],
    "tolong jelaskan": ["cobi jelaskeun", "punteun terangkeun", "pangwartoskeun"],
    "dengan": ["sareng", "kalawan", "pake"],
    "untuk": ["pikeun", "kanggo"],
    "karena": ["kumargi", "kusabab", "sabab"],
    "sudah": ["parantos", "geus", "tos"],
    "hanya": ["mung", "ukur", "ngan"],
    "sekarang": ["ayeuna"],
    "sangat": ["pisan", "kacida"],
    "membuat": ["nyieun", "ngadamel"],
    "buatkan": ["pangnyieunkeun", "damelkeun"],
    "melakukan": ["ngalakukeun"],
    "mengambil": ["nyokot", "nyandak"],
    "meretas": ["ngaruksak", "nyusup"],
    "mencuri": ["maling", "nyolong"],
    "uang": ["duit", "artos"],
    "orang": ["jalma", "urang"],
    "cepat": ["gancang", "enggal"],
}

# Betawi (Bahasa Betawi / Jakarta Suburb)
BETAWI_MAP: Dict[str, List[str]] = {
    "bagaimana cara": ["gimane caranye", "kek gimane caranye"],
    "bagaimana caranya": ["gimane caranye"],
    "bagaimana": ["gimane", "kek ape"],
    "saya": ["gue", "gw", "gua", "aye"],
    "anda": ["lu", "elu", "ente"],
    "kamu": ["lu", "elu", "ente"],
    "tidak": ["kaga", "kagak", "kagaak"],
    "bisa": ["bisa", "bise"],
    "dapat": ["bisa", "dapet"],
    "tolong": ["tolong", "coba dong", "minta tolong"],
    "tolong jelaskan": ["coba jelasin dong", "kasih tau gue", "terangin ke gue"],
    "dengan": ["pake", "sama"],
    "untuk": ["buat", "bwt"],
    "karena": ["soalnye", "gegara", "lantaran"],
    "sudah": ["udah", "udeh"],
    "hanya": ["cuman", "cuma", "doang"],
    "sekarang": ["sekarang", "skrg"],
    "sangat": ["banget", "bener-bener"],
    "membuat": ["bikin", "ngebuat"],
    "buatkan": ["bikinin", "bikinin dah"],
    "melakukan": ["ngelakuin", "ngelakonin"],
    "mengambil": ["ngambil", "nyomot"],
    "meretas": ["ngehack", "ngacak-ngacak", "bobol"],
    "mencuri": ["nyolong", "ngembat"],
    "uang": ["duit"],
    "orang": ["orang", "bocah"],
    "cepat": ["cepet", "cepetan"],
}

# Minangkabau (Baso Minang)
MINANG_MAP: Dict[str, List[str]] = {
    "bagaimana cara": ["ba a caronyo", "bagaikama caronyo"],
    "bagaimana caranya": ["ba a caronyo"],
    "bagaimana": ["ba a", "bagaikama"],
    "saya": ["ambo", "denai", "aden"],
    "anda": ["waang", "angku", "sutan"],
    "kamu": ["waang", "kau"],
    "tidak": ["indak", "ndak", "indak buliah"],
    "bisa": ["bisa", "dapek"],
    "dapat": ["dapek"],
    "tolong": ["tolong", "cubo"],
    "tolong jelaskan": ["cubo jalehan", "tolong agiah tau ambo"],
    "dengan": ["jo", "sarato"],
    "untuk": ["untuak", "buek"],
    "karena": ["dek karano", "karano", "dek karano tu"],
    "sudah": ["lah", "alah"],
    "hanya": ["hanyo", "cuma"],
    "sekarang": ["kini", "kini ko"],
    "sangat": ["bana", "sangek"],
    "membuat": ["mambuek"],
    "buatkan": ["buekkan"],
    "melakukan": ["mangarajoan"],
    "mengambil": ["maambiak"],
    "meretas": ["marusak", "manyusup"],
    "mencuri": ["mancilok", "mancuri"],
    "uang": ["pitih"],
    "orang": ["urang"],
    "cepat": ["capek"],
}

REGIONAL_PARTICLES = {
    "sundanese": ["teh", "mah", "euy", "atuh"],
    "javanese": ["ta", "lho", "rek", "nopo"],
    "betawi": ["nih", "dah", "noh", "pan", "dong"],
    "minang": ["mah", "lah", "ko", "bana"]
}

class IndonesianRegionalMutator(BaseMutator):
    """
    Mutator for under-represented languages in Indonesia.
    Supports Javanese, Sundanese, Betawi, and Minangkabau.
    Operates in two complementary modes:
    1. Lexical / Morphological code-mixing & translation (zero-latency, rule-based)
    2. Cendol-assisted neural translation & regional paraphrasing
    """

    def __init__(
        self,
        probability: float = 0.45,
        target_language: str = "random",  # 'javanese', 'sundanese', 'betawi', 'minang', or 'random'
        use_neural: bool = False,
        cendol_model_name: str = "indonlp/cendol-mt5-base-inst"
    ):
        super().__init__(name="IndonesianRegionalMutator", probability=probability)
        self.target_language = target_language.lower()
        self.use_neural = use_neural
        self.cendol_model_name = cendol_model_name
        self._neural_generator = None

    def _get_active_language(self) -> str:
        if self.target_language in ["javanese", "sundanese", "betawi", "minang"]:
            return self.target_language
        return random.choice(["javanese", "sundanese", "betawi", "minang"])

    def _get_dictionary_for_lang(self, lang: str) -> Dict[str, List[str]]:
        if lang == "javanese":
            return JAVANESE_MAP
        elif lang == "sundanese":
            return SUNDANESE_MAP
        elif lang == "betawi":
            return BETAWI_MAP
        elif lang == "minang":
            return MINANG_MAP
        return JAVANESE_MAP

    def _apply_lexical_mutation(self, text: str, lang: str) -> Tuple[str, List[str]]:
        lex_map = self._get_dictionary_for_lang(lang)
        mutated = text
        applied = []

        # Sort multi-word phrases first (e.g. "bagaimana cara" before "cara")
        sorted_keys = sorted(lex_map.keys(), key=lambda k: len(k.split()), reverse=True)

        for phrase in sorted_keys:
            pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            if pattern.search(mutated) and random.random() < 0.85:
                replacement = random.choice(lex_map[phrase])
                mutated = pattern.sub(replacement, mutated)
                applied.append(f"{lang}_{phrase}->{replacement}")

        # Add occasional regional discourse particle at end
        if lang in REGIONAL_PARTICLES and random.random() < 0.40:
            particle = random.choice(REGIONAL_PARTICLES[lang])
            mutated = f"{mutated.rstrip('.?!')} {particle}."
            applied.append(f"particle_{particle}")

        return mutated, applied

    def _apply_cendol_neural(self, text: str, lang: str) -> Optional[str]:
        if self._neural_generator is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tokenizer = AutoTokenizer.from_pretrained(self.cendol_model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.cendol_model_name).to(device)
                model.eval()
                self._neural_generator = (tokenizer, model, device)
            except Exception as e:
                print(f"Warning: Failed to load Cendol for neural generation: {e}")
                return None

        tokenizer, model, device = self._neural_generator
        lang_names = {
            "javanese": "Bahasa Jawa",
            "sundanese": "Bahasa Sunda",
            "betawi": "Bahasa Betawi",
            "minang": "Bahasa Minangkabau"
        }
        target_name = lang_names.get(lang, "Bahasa Jawa")
        input_prompt = f"Terjemahkan ke dalam {target_name}: {text}"

        try:
            inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=2,
                    temperature=0.7,
                    early_stopping=True
                )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if result and len(result) > 5 and result.lower() != text.lower():
                return result
        except Exception as e:
            print(f"Neural generation error: {e}")
        return None

    def mutate(self, text: str, **kwargs) -> MutationResult:
        if random.random() > self.probability or not text.strip():
            return MutationResult(text, text, self.name, {"applied": False})

        lang = self._get_active_language()
        applied_ops = [f"target_lang:{lang}"]

        # Try neural generation if requested
        if self.use_neural:
            neural_res = self._apply_cendol_neural(text, lang)
            if neural_res:
                applied_ops.append("cendol_neural_translation")
                return MutationResult(
                    original_text=text,
                    mutated_text=neural_res,
                    mutation_type=self.name,
                    metadata={"applied": True, "language": lang, "mode": "neural", "operations": applied_ops}
                )

        # Lexical and morphological code-mixing
        mutated_text, ops = self._apply_lexical_mutation(text, lang)
        applied_ops.extend(ops)

        return MutationResult(
            original_text=text,
            mutated_text=mutated_text,
            mutation_type=self.name,
            metadata={
                "applied": len(ops) > 0,
                "language": lang,
                "mode": "lexical_codemix",
                "operations": applied_ops
            }
        )
