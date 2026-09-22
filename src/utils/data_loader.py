"""
Data loading and preprocessing utilities for Indonesian LLM safety benchmarks and jailbreak seeds.
Supports IndoSafety benchmark formats and localized datasets.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

class BenchmarkDataLoader:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.raw_dir = self.data_dir / "raw"
        self.localized_dir = self.data_dir / "localized"
        self.templates_dir = self.data_dir / "templates"

    def load_seed_templates(self, filename: str = "seeds_id.json") -> List[Dict[str, Any]]:
        """Load Indonesian seed jailbreak templates."""
        template_file = self.templates_dir / filename
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found at: {template_file}")
        with open(template_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_localized_dataset(self, filename: str = "indosafety_eval1_all.json") -> List[Dict[str, Any]]:
        """Load parsed & localized evaluation dataset."""
        dataset_file = self.localized_dir / filename
        if not dataset_file.exists():
            raise FileNotFoundError(f"Localized dataset not found at: {dataset_file}")
        with open(dataset_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_parallel_dataset(self, filename: str = "indosafety_eval2_parallel.json") -> List[Dict[str, Any]]:
        """Load parallel regional & colloquial evaluation dataset (Eval-2)."""
        dataset_file = self.localized_dir / filename
        if not dataset_file.exists():
            raise FileNotFoundError(f"Parallel dataset not found at: {dataset_file}")
        with open(dataset_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_prompts_by_variety(
        self, 
        records: List[Dict[str, Any]], 
        sheet_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Filter prompts by sheet or language variety (e.g., formal vs colloquial Indonesian)."""
        if not sheet_name:
            return records
        return [r for r in records if r.get("sheet") == sheet_name]

    def get_prompts_by_dialect(
        self,
        records: List[Dict[str, Any]],
        dialect: str = "formal"
    ) -> List[Dict[str, Any]]:
        """
        Extract prompt records tailored to a specific dialect/variety from Eval-2.
        Dialect options: 'formal', 'colloquial', 'java', 'sunda', 'minangkabau'
        """
        dialect_key = dialect.lower()
        results = []
        for r in records:
            varieties = r.get("varieties", {})
            prompt_text = varieties.get(dialect_key)
            if prompt_text:
                item = {
                    "id": r.get("id"),
                    "risk_area": r.get("risk_area"),
                    "types_of_harm": r.get("types_of_harm"),
                    "specific_harms": r.get("specific_harms"),
                    "dialect": dialect_key,
                    "prompt": prompt_text,
                    "source": r.get("source", "")
                }
                results.append(item)
        return results

    def get_prompts_by_risk_area(
        self,
        records: List[Dict[str, Any]],
        risk_area: str
    ) -> List[Dict[str, Any]]:
        """Filter records by specific risk area (case-insensitive substring match)."""
        target = risk_area.lower()
        results = []
        for r in records:
            area = r.get("risk_area", "")
            if not area and "data" in r:
                area = r["data"].get("risk_area", "")
            if target in str(area).lower():
                results.append(r)
        return results

    def sample_prompts(
        self,
        records: List[Dict[str, Any]],
        n: int = 10,
        seed: Optional[int] = 42
    ) -> List[Dict[str, Any]]:
        """Deterministically sample n records."""
        import random
        if seed is not None:
            rng = random.Random(seed)
            return rng.sample(records, min(n, len(records)))
        return records[:n]
