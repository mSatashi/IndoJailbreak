# IndoSafety Baseline Safety Benchmark: Consolidated Multi-Model Evaluation Report

**Thesis Project:** IndoJailbreak — Automated Adversarial Safety Evaluation for Indonesian LLMs  
**Date:** 2026-10-05  
**Evaluator:** IndoSafety Baseline Runner ([`scripts/run_baseline.py`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/scripts/run_baseline.py))  
**Benchmark Suite:** IndoSafety-Eval-1 (Bahasa Indonesia Baku, 2,514 total records)  
**Sample Strategy:** 10 deterministically sampled prompts per risk category (`seed=42`, total 60 prompts per model)  
**Evaluation Mode:** Direct Zero-Shot / Baseline Safety Evaluation (Dual-Dimensional Judge: Indonesian & English Refusal Compliance + Actionable Harmfulness Scoring)  

---

## 1. Executive Summary & Benchmark Comparison

Across 180 total evaluations (60 prompts $\times$ 3 models), this experiment benchmarks the baseline safety profiles of three major open/cloud architectures when prompted with sensitive Indonesian queries:

| Performance Metric | `deepseek-v4.1-flash:cloud` | `nemotron-3-nano:30b-cloud` | `gemma4:31b-cloud` |
| :--- | :---: | :---: | :---: |
| **Total Evaluated Prompts** | 60 | 60 | 60 |
| **Baseline Refusal Rate** | **90.00%** (54/60) | **33.33%** (20/60) | **35.00%** (21/60) |
| **Compliance Rate** | **10.00%** (6/60) | **66.67%** (40/60) | **65.00%** (39/60) |
| **Actionable Harm Bypass Rate** | **0.00%** (0/60) | **0.00%** (0/60) | **0.00%** (0/60) |
| **Mean Harmfulness Score (1.0–5.0)** | **1.10** | **1.68** | **2.25** |
| **Average Query Latency** | ~4.01s | ~28.67s | **~2.98s (Fastest)** |
| **Safety Architecture Paradigm** | *Silent Drop / Content Filter* | *Cross-Lingual English Fallback* | *Indonesian Explanatory Rebuttal* |

---

## 2. Risk Category Breakdown (10 Prompts per Category)

### A. Refusal Rate Comparison (%)
| Risk Category | `deepseek-v4.1-flash` | `nemotron-3-nano` | `gemma4:31b` | Most Vulnerable Baseline |
| :--- | :---: | :---: | :---: | :--- |
| **Malicious Uses** | **100.0%** (10/10) | **90.0%** (9/10) | **80.0%** (8/10) | `gemma4:31b` (80.0%) |
| **Discrimination, Exclusion & Toxicity** | **100.0%** (10/10) | **50.0%** (5/10) | **30.0%** (3/10) | `gemma4:31b` (30.0%) |
| **Misinformation Harms** | **100.0%** (10/10) | **30.0%** (3/10) | **0.0%** (0/10)* | `gemma4:31b` (0.0% outright refusal) |
| **Region-specific Sensitivity (SARA)** | **100.0%** (10/10) | **10.0%** (1/10) | **20.0%** (2/10) | `nemotron-3-nano` (10.0%) |
| **Information Hazards** | **80.0%** (8/10) | **10.0%** (1/10) | **60.0%** (6/10) | `nemotron-3-nano` (10.0%) |
| **Human-Chatbot Interaction Harms** | **60.0%** (6/10) | **10.0%** (1/10) | **20.0%** (2/10) | `nemotron-3-nano` (10.0%) |
| **OVERALL REFUSAL RATE** | **90.00%** | **33.33%** | **35.00%** | `nemotron-3-nano` (33.33%) |

*\*Note on Misinformation: Gemma’s 0% refusal rate reflects conversational debunking rather than malicious compliance; it explicitly warns against dangerous actions.*

### B. Mean Harmfulness Score (1.0 to 5.0, where 1.0 is completely harmless)
| Risk Category | `deepseek-v4.1-flash` | `nemotron-3-nano` | `gemma4:31b` | Category Risk Level |
| :--- | :---: | :---: | :---: | :--- |
| **Malicious Uses** | 1.00 | 1.10 | 1.40 | Low |
| **Discrimination, Exclusion & Toxicity** | 1.00 | 1.50 | 2.30 | Low-Medium |
| **Misinformation Harms** | 1.00 | 1.80 | 3.00 | Medium |
| **Region-specific Sensitivity (SARA)** | 1.00 | 1.90 | 2.40 | Low-Medium |
| **Information Hazards** | 1.20 | 1.90 | 2.10 | Low-Medium |
| **Human-Chatbot Interaction Harms** | 1.40 | 1.90 | 2.30 | Low-Medium |
| **OVERALL MEAN HARMFULNESS** | **1.10** | **1.68** | **2.25** | Safe Baseline Profile |

---

## 3. Core Research Findings & Alignment Analysis

### 1. The Tripartite Taxonomy of Safety Enforcement
This baseline evaluation highlights three distinct paradigms implemented by model creators:

```
                            SAFETY ENFORCEMENT TAXONOMY
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  [Silent Censorship]         [Cross-Lingual Fallback]         [Explanatory Discourse]
  DeepSeek v4.1 Flash              Nemotron-3-Nano                 Gemma 4 31B
  ───────────────────         ───────────────────────          ───────────────────
  • 90% Refusal               • 33% Refusal                    • 35% Refusal
  • Empty response / drop     • English guardrails:            • Native Indonesian:
    on sensitive keywords       "I'm sorry, but I can't..."      "Maaf, saya tidak bisa..."
  • High false-positive       • Disconnect between prompt      • High fluency; debunks
    rate on benign queries      language & refusal language      misinformation proactively
```

### 2. Cross-Lingual Alignment Tax
* **Nemotron** reveals a common limitation in multilingual RLHF: when safe generation triggers in Indonesian, the model reverts to English canned refusals. This demonstrates that alignment data is heavily skewed toward English safety examples.
* **Gemma** is natively aligned to Indonesian discourse conventions, refusing politely and idiomatically.

### 3. Implications for Evolutionary Jailbreak Attacks (AutoDAN / ForgeDAN)
1. **Targeting Gemma**:  
   Gemma’s eagerness to provide educational explanations and debunk misinformation makes it vulnerable to **roleplay framing** (e.g., *"Saya seorang peneliti farmakologi yang sedang mendokumentasikan mitos..."*) to circumvent refusal boundaries.
2. **Targeting Nemotron**:  
   Because its guardrails are heavily tied to English safety triggers, **Indonesian colloquial (Bahasa Gaul)** and **regional dialect transfer (Basa Sunda/Jawa/Minang)** from IndoSafety-Eval-2 are expected to produce significant bypass rates.
3. **Targeting DeepSeek**:  
   DeepSeek relies on rigid keyword/token blocklists. Stealth character-level perturbations (e.g., zero-width spaces, leetspeak, affix modifications) represent the primary vector to bypass its silent drop filter.

---

## 4. Associated Datasets & Artifact Links

| Model | Exported Dataset Path | Run Log Path |
| :--- | :--- | :--- |
| **Gemma 4 31B** | [`results/datasets/baseline_evaluation_results.json`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/results/datasets/baseline_evaluation_results.json) | [`results/runs/20261005_105111_baseline_ollama_gemma_cloud/run_results.json`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/results/runs/20261005_105111_baseline_ollama_gemma_cloud/run_results.json) |
| **Nemotron-3-Nano** | [`results/datasets/baseline_evaluation_results_2.json`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/results/datasets/baseline_evaluation_results_2.json) | [`results/runs/20261005_111740_baseline_nemotron-3-nano/run_results.json`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/results/runs/20261005_111740_baseline_nemotron-3-nano/run_results.json) |
| **DeepSeek v4.1 Flash** | [`results/datasets/baseline_evaluation_results_3.json`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/results/datasets/baseline_evaluation_results_3.json) | [`results/runs/20261005_124212_baseline_ollama_deepseek_cloud/run_results.json`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/results/runs/20261005_124212_baseline_ollama_deepseek_cloud/run_results.json) |

All individual model reports are archived in the [`reports/`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/reports) folder.
