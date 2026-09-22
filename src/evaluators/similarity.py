"""
Semantic similarity evaluators for Indonesian LLM jailbreaks.
Supports IndoNLP's Cendol model family (e.g., indonlp/cendol-mt5-base-inst)
as well as Sentence-Transformers for semantic preservation and fitness scoring.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional
import torch
import torch.nn.functional as F

class BaseSemanticEvaluator(ABC):
    """Abstract interface for semantic similarity evaluation."""

    @abstractmethod
    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity score between two texts in range [0.0, 1.0]."""
        pass

    @abstractmethod
    def batch_compute_similarity(self, reference: str, candidates: List[str]) -> List[float]:
        """Compute similarity between a single reference goal and a batch of candidate prompts."""
        pass


class CendolSemanticEvaluator(BaseSemanticEvaluator):
    """
    Evaluator using IndoNLP's Cendol model family.
    Extracts semantic representations via Cendol's encoder hidden states
    with Indonesian vocabulary adaptation, then computes cosine similarity.
    
    Default model: 'indonlp/cendol-mt5-base-inst'
    Alternative: 'indonlp/cendol-mt5-small-chat', 'indonlp/cendol-mt5-large-chat'
    """

    def __init__(
        self,
        model_name: str = "indonlp/cendol-mt5-base-inst",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

    def _lazy_load(self):
        """Loads Cendol model on first use to conserve memory until needed."""
        if self._is_loaded:
            return

        from transformers import AutoTokenizer, AutoModel

        print(f"Loading Cendol model: {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load encoder for representation extraction
        try:
            from transformers import MT5EncoderModel
            self.model = MT5EncoderModel.from_pretrained(self.model_name).to(self.device)
        except Exception:
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        self.model.eval()
        self._is_loaded = True
        print(f"Cendol model {self.model_name} loaded successfully.")

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Attention-mask weighted mean pooling."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Generates normalized L2 embeddings for a list of texts."""
        self._lazy_load()
        
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Use last_hidden_state from encoder
            last_hidden = model_output.last_hidden_state
            pooled = self._mean_pooling(last_hidden, encoded_input["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)

        return normalized

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Computes cosine similarity between text_a and text_b."""
        embs = self.encode([text_a, text_b])
        cos_sim = torch.dot(embs[0], embs[1]).item()
        # Clamp to [0.0, 1.0] for clean fitness scoring
        return float(max(0.0, min(1.0, (cos_sim + 1.0) / 2.0)))

    def batch_compute_similarity(self, reference: str, candidates: List[str]) -> List[float]:
        """Computes similarity of all candidates against a reference goal."""
        if not candidates:
            return []

        all_texts = [reference] + candidates
        embs = self.encode(all_texts)
        
        ref_emb = embs[0:1]         # Shape: [1, D]
        cand_embs = embs[1:]        # Shape: [N, D]

        cos_sims = torch.mm(cand_embs, ref_emb.t()).squeeze(-1)
        # Scale to [0.0, 1.0]
        scaled = (cos_sims + 1.0) / 2.0
        clamped = torch.clamp(scaled, 0.0, 1.0)
        return clamped.cpu().tolist()


class SentenceTransformerEvaluator(BaseSemanticEvaluator):
    """
    Fallback evaluator using sentence-transformers (e.g., LazarusNLP/all-indo-e5-small-v2
    or paraphrase-multilingual-mpnet-base-v2).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._is_loaded = False

    def _lazy_load(self):
        if self._is_loaded:
            return
        from sentence_transformers import SentenceTransformer
        print(f"Loading SentenceTransformer: {self.model_name} on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self._is_loaded = True

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        self._lazy_load()
        embs = self.model.encode([text_a, text_b], convert_to_tensor=True, normalize_embeddings=True)
        cos_sim = torch.dot(embs[0], embs[1]).item()
        return float(max(0.0, min(1.0, (cos_sim + 1.0) / 2.0)))

    def batch_compute_similarity(self, reference: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        self._lazy_load()
        ref_emb = self.model.encode(reference, convert_to_tensor=True, normalize_embeddings=True)
        cand_embs = self.model.encode(candidates, convert_to_tensor=True, normalize_embeddings=True)
        cos_sims = torch.mv(cand_embs, ref_emb)
        scaled = (cos_sims + 1.0) / 2.0
        return torch.clamp(scaled, 0.0, 1.0).cpu().tolist()


def get_semantic_evaluator(
    evaluator_type: str = "cendol",
    model_name: Optional[str] = None,
    device: Optional[str] = None
) -> BaseSemanticEvaluator:
    """Factory helper to instantiate semantic evaluators."""
    if evaluator_type.lower() == "cendol":
        name = model_name or "indonlp/cendol-mt5-base-inst"
        return CendolSemanticEvaluator(model_name=name, device=device)
    else:
        name = model_name or "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        return SentenceTransformerEvaluator(model_name=name, device=device)
