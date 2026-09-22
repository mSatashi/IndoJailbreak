# IndoJailbreak: Automated Adversarial Safety Evaluation for Indonesian LLMs

An evolutionary adversarial jailbreak framework inspired by **AutoDAN** and **ForgeDAN**, specifically designed to evaluate and red-team Large Language Models in **Bahasa Indonesia**.

---

## 🔬 Motivation & Research Scope
* **Cross-Lingual Alignment Tax**: Most safety alignment (RLHF/DPO) in frontier LLMs is optimized for English, leaving significant safety vulnerabilities in Indonesian prompts.
* **Beyond Literal Translation**: Naive translation of English jailbreaks misses Indonesian morphology (affixes), diglossia (formal *bahasa baku* vs. colloquial *bahasa gaul*), and regional cultural contexts.
* **Core Principles**:
  * **AutoDAN**: Stealthy, interpretable, genetic algorithm-based adversarial prompt search.
  * **ForgeDAN**: Multi-strategy hierarchical perturbations (character, word, and template levels), semantic similarity constraints, and dual-dimensional judgment (refusal compliance vs. harmfulness).

---

## 📁 Repository Structure

```
Tesis_Project/
├── configs/
│   ├── default_config.yaml       # Genetic search hyperparameters & weights
│   └── models.yaml               # Target models (HuggingFace / Ollama / APIs) & judges
├── data/
│   ├── raw/
│   │   └── indosafety/           # IndoSafety benchmark Excel files, rubrics & templates
│   ├── localized/                # Processed benchmark JSON files (2,514 items)
│   └── templates/
│       └── seeds_id.json         # Indonesian jailbreak seed templates
├── scripts/
│   ├── download_indosafety.py    # Automated IndoSafety benchmark fetcher & parser (Eval-1 & Eval-2)
│   └── run_attack.py             # CLI experiment runner & adversarial dataset generator
├── src/
│   ├── core/                     # Evolutionary optimization engine (AutoDAN / ForgeDAN)
│   ├── mutators/                 # Char-, word-, regional- (Jawa/Sunda/Minang/Betawi), and template-level mutators
│   ├── evaluators/               # Semantic similarity & fluency evaluators (IndoNLP Cendol / Sentence-Transformers)
│   ├── judges/                   # Refusal compliance (baku/gaul/regional) & harmfulness scoring
│   ├── models/                   # Unified target LLM interface (HF 4-bit, Ollama, OpenAI, Gemini)
│   └── utils/
│       └── data_loader.py        # Benchmark & template data loader
├── tests/
│   ├── test_data_loader.py       # Benchmark (Eval-1 & Eval-2) and loader test
│   ├── test_mutators.py          # Indonesian char, word, template mutator tests
│   ├── test_regional_mutators.py # Regional language mutator tests
│   ├── test_evaluators.py        # Semantic similarity evaluator tests
│   ├── test_models.py            # Unified target LLM interface test
│   └── test_genetic_engine.py    # End-to-end evolutionary search loop test
├── results/
│   ├── datasets/                 # Exported Indonesian Adversarial Benchmark Datasets
│   └── runs/                     # Detailed run logs, metrics, and generation histories
└── requirements.txt
```

---

## 📊 Benchmark Datasets: IndoSafety (Eval-1 & Eval-2)
This project utilizes the **IndoSafety** benchmark suite (Falensi Azmi et al., EMNLP 2025):
* **IndoSafety-Eval-1**: 2,514 raw safety evaluation prompts across 6 risk domains.
* **IndoSafety-Eval-2**: 500 parallel safety prompts across 5 language varieties:
  * `indonesian-formal` (*Bahasa Baku*)
  * `colloquial` (*Bahasa Gaul / Slang*)
  * `java` (*Basa Jawa*)
  * `sunda` (*Basa Sunda*)
  * `minangkabau` (*Baso Minang*)

---

## 🚀 Getting Started & Execution

### 1. Run Unit Tests
```powershell
python tests/test_data_loader.py
python tests/test_regional_mutators.py
python tests/test_models.py
python tests/test_genetic_engine.py
```

### 2. Execute Adversarial Safety Evaluation (CLI)
```powershell
# Run dry-run test with mock model across all 5 dialects
python scripts/run_attack.py --target-model mock --num-samples 5 --dataset eval2 --dialect all

# Evaluate Ollama local target (e.g. Llama-3)
python scripts/run_attack.py --target-model ollama_llama3 --num-samples 10 --dataset eval2 --dialect colloquial

# Evaluate HuggingFace 4-bit local model (e.g. SEA-LION or Llama-3)
python scripts/run_attack.py --target-model sealion_7b --num-samples 10 --dataset eval2 --dialect java

# Evaluate OpenAI target (GPT-4o-mini)
python scripts/run_attack.py --target-model gpt_4o_mini --num-samples 10 --dataset eval2 --dialect formal

# Evaluate OpenRouter target (e.g. Llama-3.3-70B, Qwen-2.5-72B, or DeepSeek)
$env:OPENROUTER_API_KEY="your_openrouter_api_key"
python scripts/run_attack.py --target-model openrouter_llama33_70b --num-samples 10 --dataset eval2 --dialect colloquial
```

### 3. Generated Adversarial Benchmark Dataset
All experiments automatically export the resulting adversarial dataset to:
`results/datasets/indojailbreak_adversarial_bench.json`
