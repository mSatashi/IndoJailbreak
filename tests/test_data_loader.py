import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_loader import BenchmarkDataLoader

def test_data_loader():
    loader = BenchmarkDataLoader()
    
    # 1. Test loading seed templates
    templates = loader.load_seed_templates()
    print(f"Loaded {len(templates)} seed templates.")
    assert len(templates) >= 5, "Should have at least 5 seed templates"
    for t in templates:
        assert "id" in t and "template" in t
        print(f"  - Template: {t['id']} ({t['name']})")

    # 2. Test loading localized IndoSafety benchmark
    records = loader.load_localized_dataset()
    print(f"\nLoaded {len(records)} benchmark records from IndoSafety-Eval-1.")
    assert len(records) > 0, "Benchmark should not be empty"
    
    first_record = records[0]["data"]
    print("\nSample Record Keys:", list(first_record.keys()))
    print(f"Sample ID: {first_record.get('id')}")
    print(f"Sample Risk Area: {first_record.get('risk_area')}")
    print(f"Sample Type of Harm: {first_record.get('types_of_harm')}")
    print(f"Sample Prompt snippet: {str(first_record.get('prompt'))[:80]}...")
    
    # Check risk areas
    risk_areas = set(r["data"].get("risk_area") for r in records if "risk_area" in r["data"])
    print(f"\nDistinct Risk Areas in Benchmark ({len(risk_areas)}):")
    for area in sorted(str(a) for a in risk_areas):
        print(f"  * {area}")

    # 3. Test loading parallel dialect dataset (IndoSafety-Eval-2)
    parallel_records = loader.load_parallel_dataset()
    print(f"\nLoaded {len(parallel_records)} parallel dialect records from IndoSafety-Eval-2.")
    assert len(parallel_records) == 500, "Should have exactly 500 parallel records"

    # 4. Test dialect extraction
    dialects = ["formal", "colloquial", "java", "sunda", "minangkabau"]
    for d in dialects:
        d_prompts = loader.get_prompts_by_dialect(parallel_records, dialect=d)
        assert len(d_prompts) == 500, f"Each dialect should have 500 entries, got {len(d_prompts)} for {d}"
        print(f"  - [{d:11}] Sample: {d_prompts[0]['prompt'][:50]}...")

    # 5. Test sampling and risk area filtering
    toxicity_prompts = loader.get_prompts_by_risk_area(parallel_records, "Toxicity")
    print(f"\nFiltered {len(toxicity_prompts)} records for risk area 'Toxicity'.")
    assert len(toxicity_prompts) > 0, "Should find records for Toxicity"

    sample = loader.sample_prompts(parallel_records, n=5)
    assert len(sample) == 5, "Sample should contain exactly 5 records"
    print("DataLoader tests completed successfully!")

if __name__ == "__main__":
    test_data_loader()
