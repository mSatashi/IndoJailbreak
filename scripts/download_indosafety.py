"""
Utility script to fetch and prepare the IndoSafety benchmark dataset.
Source: Falensi Azmi et al., "IndoSafety: Culturally Grounded Safety for LLMs in Indonesian Languages" (EMNLP 2025).
Repository: https://github.com/falensiazmi/IndoSafety
"""

import os
import urllib.request
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "indosafety"
LOCALIZED_DIR = BASE_DIR / "data" / "localized"
TEMPLATES_DIR = BASE_DIR / "data" / "templates"

FILES_TO_DOWNLOAD = {
    "IndoSafety-Eval-1.xlsx": "https://raw.githubusercontent.com/falensiazmi/IndoSafety/main/dataset/IndoSafety-Eval-1.xlsx",
    "IndoSafety-Eval-2.xlsx": "https://raw.githubusercontent.com/falensiazmi/IndoSafety/main/dataset/IndoSafety-Eval-2.xlsx",
    "rubrics_general.xlsx": "https://raw.githubusercontent.com/falensiazmi/IndoSafety/main/eval_rubric/rubrics_general.xlsx",
    "rubrics_indonesia.xlsx": "https://raw.githubusercontent.com/falensiazmi/IndoSafety/main/eval_rubric/rubrics_indonesia.xlsx",
    "eval_template.txt": "https://raw.githubusercontent.com/falensiazmi/IndoSafety/main/prompt_templates/eval_template.txt",
}

def download_raw_files():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOCALIZED_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading raw IndoSafety files into {RAW_DIR}...")
    for filename, url in FILES_TO_DOWNLOAD.items():
        dest_path = RAW_DIR / filename
        if not dest_path.exists():
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, dest_path)
            print(f"  Saved to {dest_path}")
        else:
            print(f"  {filename} already exists, skipping download.")

def extract_and_localize():
    try:
        import pandas as pd
    except ImportError:
        print("pandas not found. Please install pandas and openpyxl first.")
        return

    eval1_path = RAW_DIR / "IndoSafety-Eval-1.xlsx"
    if not eval1_path.exists():
        print(f"File {eval1_path} not found.")
        return

    print("Extracting evaluation sets from IndoSafety...")
    excel_file = pd.ExcelFile(eval1_path)
    print(f"Sheets in {eval1_path.name}: {excel_file.sheet_names}")

    all_prompts = []
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(eval1_path, sheet_name=sheet_name)
        print(f"Sheet '{sheet_name}' shape: {df.shape}, columns: {list(df.columns)}")
        
        # Standardize records
        for idx, row in df.iterrows():
            record = {
                "source_dataset": "IndoSafety-Eval-1",
                "sheet": sheet_name,
                "row_index": idx,
                "data": row.dropna().to_dict()
            }
            all_prompts.append(record)

    output_json = LOCALIZED_DIR / "indosafety_eval1_all.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, ensure_ascii=False, indent=2)
    print(f"Successfully exported {len(all_prompts)} raw items to {output_json}")

    # Extract Eval-2 (Parallel Dialects)
    eval2_path = RAW_DIR / "IndoSafety-Eval-2.xlsx"
    if not eval2_path.exists():
        print(f"File {eval2_path} not found.")
        return

    print("\nExtracting parallel dialect evaluation set (Eval-2)...")
    df_eval2 = pd.read_excel(eval2_path)
    print(f"Eval-2 shape: {df_eval2.shape}, columns: {list(df_eval2.columns)}")

    eval2_records = []
    for idx, row in df_eval2.iterrows():
        record = {
            "id": int(row.get("id", idx)) if pd.notna(row.get("id")) else idx,
            "risk_area": str(row.get("risk_area", "")),
            "types_of_harm": str(row.get("types_of_harm", "")),
            "specific_harms": str(row.get("specific_harms", "")),
            "source": str(row.get("source", "")),
            "varieties": {
                "formal": str(row.get("indonesian-formal", "")).strip(),
                "colloquial": str(row.get("colloquial", "")).strip(),
                "minangkabau": str(row.get("minangkabau", "")).strip(),
                "java": str(row.get("java", "")).strip(),
                "sunda": str(row.get("sunda", "")).strip(),
            }
        }
        eval2_records.append(record)

    output_eval2_json = LOCALIZED_DIR / "indosafety_eval2_parallel.json"
    with open(output_eval2_json, "w", encoding="utf-8") as f:
        json.dump(eval2_records, f, ensure_ascii=False, indent=2)
    print(f"Successfully exported {len(eval2_records)} parallel dialect items to {output_eval2_json}")

if __name__ == "__main__":
    download_raw_files()
    extract_and_localize()
