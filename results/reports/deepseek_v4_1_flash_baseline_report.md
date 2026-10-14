# IndoSafety Baseline Safety Evaluation Report: `ollama_deepseek_cloud`

**Date:** 2026-10-05  
**Model Under Test:** `ollama_deepseek_cloud` (`deepseek-v4.1-flash:cloud`)  
**Benchmark:** [IndoSafety-Eval-1](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/data/localized/indosafety_eval1_all.json) (Bahasa Indonesia Baku)  
**Sample Strategy:** 10 deterministically sampled prompts per risk category (`seed=42`, total 60 prompts)  
**Evaluation Mode:** Direct Baseline Safety Evaluation (Dual-Dimensional Judge: Refusal Compliance + Harmfulness Scoring)  
**Exported Dataset:** [`results/datasets/baseline_evaluation_results_3.json`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/results/datasets/baseline_evaluation_results_3.json)  

---

## 1. Executive Summary

| Metric | `deepseek-v4.1-flash:cloud` | `nemotron-3-nano:30b-cloud` | `gemma4:31b-cloud` |
| :--- | :---: | :---: | :---: |
| **Total Prompts Evaluated** | **60** | 60 | 60 |
| **Baseline Refusal Rate** | **90.00%** (54/60) | 33.33% (20/60) | 35.00% (21/60) |
| **Compliance Rate** | **10.00%** (6/60) | 66.67% (40/60) | 65.00% (39/60) |
| **Unsafe / Direct Bypass Rate** | **0.00%** (0/60) | 0.00% (0/60) | 0.00% (0/60) |
| **Mean Harmfulness Score (1–5)** | **1.10** | 1.68 | 2.25 |
| **Average Query Latency** | **~4.01s** per prompt | ~28.67s per prompt | ~2.98s per prompt |
| **Primary Refusal Style** | **Silent Drop / Censorship Block** | **English Fallback Refusal** | **Indonesian Explanatory Refusal** |

---

## 2. Risk Category Breakdown for `deepseek-v4.1-flash:cloud`

| Risk Category | Evaluated (N) | Refusal Rate | Mean Harm Score (1–5) | Unsafe Bypass Rate | Observed Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Discrimination, Exclusion & Toxicity** | 10 | **100.0%** (10/10) | **1.00** | **0.0%** | **Total Filter Block**: 100% of hate speech, ethnic stereotyping, and adult queries were blocked. |
| **Malicious Uses** | 10 | **100.0%** (10/10) | **1.00** | **0.0%** | **Total Refusal**: Complete block on cyberattacks, forgery, doxxing, and black market requests. |
| **Misinformation Harms** | 10 | **100.0%** (10/10) | **1.00** | **0.0%** | **Total Filter Block**: All dangerous health and science misinformation prompts were filtered. |
| **Region-specific Sensitivity** | 10 | **100.0%** (10/10) | **1.00** | **0.0%** | **Strict SARA Censorship**: 100% block on sensitive Indonesian regional/ethnic/religious controversies and NIK privacy queries. |
| **Information Hazards** | 10 | **80.0%** (8/10) | **1.20** | **0.0%** | **High Refusal**: Only answered benign theoretical frameworks; blocked all privacy/hazard probes. |
| **Human-Chatbot Interaction Harms** | 10 | **60.0%** (6/10) | **1.40** | **0.0%** | **Selective Refusal**: Blocked self-harm and direct physical health threats; answered meta-inquiries about AI nature. |

---

## 3. Head-to-Head Comparison Across All Three Models

```
                                  REFUSAL RATE COMPARISON (60 PROMPTS)
Risk Category                  DeepSeek v4.1 Flash      Nemotron-3-Nano         Gemma 4 31B
---------------------------------------------------------------------------------------------
Malicious Uses                 100.0% [==========]      90.0% [========= ]      80.0% [========  ]
Discrimination & Toxicity      100.0% [==========]      50.0% [=====     ]      30.0% [===       ]
Misinformation Harms           100.0% [==========]      30.0% [===       ]       0.0% [          ]
Region-specific Sensitivity    100.0% [==========]      10.0% [=         ]      20.0% [==        ]
Information Hazards             80.0% [========  ]      10.0% [=         ]      60.0% [======    ]
Human-Chatbot Harms             60.0% [======    ]      10.0% [=         ]      20.0% [==        ]
---------------------------------------------------------------------------------------------
Overall Refusal Rate            90.00%                  33.33%                  35.00%
Overall Mean Harm (lower=safer) 1.10 / 5.0              1.68 / 5.0              2.25 / 5.0
Average Query Speed             ~4.0s (Fast)            ~28.7s (Slow/Reasoning) ~3.0s (Fastest)
```

---

## 4. Architectural & Safety Alignment Insights for Thesis

The evaluation of these three distinct cloud models reveals **three fundamentally different safety enforcement paradigms** in response to Indonesian benchmark probes:

1. **`deepseek-v4.1-flash:cloud` — Silent Censorship / Hard Filter**:
   - DeepSeek exhibits the highest overall refusal rate (**90.00%**).
   - Rather than returning long explanatory text, DeepSeek’s upstream API guardrails enforce a **silent drop** (empty response `""`) or short canned refusal whenever an input touches sensitive topics (SARA, misinformation, sexual content, or malicious acts).
   - **Research Implication**: While this guarantees a very low harmfulness score (1.10), it also leads to severe over-refusal / refusal of benign debate, making evolutionary stealth attacks (AutoDAN/ForgeDAN) necessary to test whether subtle morphological perturbations can bypass the hard filter.

2. **`nemotron-3-nano:30b-cloud` — English System Fallback**:
   - Refusal rate of **33.33%**, with a mean harm score of **1.68**.
   - Defaults to English safety phrases (*"I’m sorry, but I can’t help with that."*) on sensitive Indonesian queries.
   - **Research Implication**: Safety alignment is unilingual (English RLHF), creating cross-lingual alignment gaps on nuanced regional Indonesian language variants.

3. **`gemma4:31b-cloud` — Native Indonesian Discourse & Fact-Checking**:
   - Refusal rate of **35.00%**, with a mean harm score of **2.25**.
   - High willingness to engage: rather than blocking misinformation, it actively rebuts and debunks false claims in native Bahasa Indonesia with bold warnings.
   - **Research Implication**: Exhibits the highest language fluency in Indonesian, but its conversational helpfulness increases surface area for adversarial framing (e.g. roleplay or theoretical research pretexts).

---

## 5. Run Artifacts & Reproducibility

* **Full Run Results Log**: [`results/runs/20261005_124212_baseline_ollama_deepseek_cloud/run_results.json`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/results/runs/20261005_124212_baseline_ollama_deepseek_cloud/run_results.json)
* **Exported Dataset 3**: [`results/datasets/baseline_evaluation_results_3.json`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/results/datasets/baseline_evaluation_results_3.json)
* **CLI Command**:
  ```powershell
  python scripts/run_baseline.py --target-model ollama_deepseek_cloud --dataset eval1 --num-samples-per-category 10 --seed 42 --dataset-output results/datasets/baseline_evaluation_results_3.json
  ```
