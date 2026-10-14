# IndoSafety Baseline Safety Evaluation Report: `nemotron-3-nano`

**Date:** 2026-10-05  
**Model Under Test:** `nemotron-3-nano` (`nemotron-3-nano:30b-cloud`)  
**Benchmark:** [IndoSafety-Eval-1](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/data/localized/indosafety_eval1_all.json) (Bahasa Indonesia Baku)  
**Sample Strategy:** 10 deterministically sampled prompts per risk category (`seed=42`, total 60 prompts)  
**Evaluation Mode:** Direct Baseline Safety Evaluation (Dual-Dimensional Judge: Refusal Compliance + Harmfulness Scoring)  
**Exported Dataset:** [`results/datasets/baseline_evaluation_results_2.json`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/results/datasets/baseline_evaluation_results_2.json)  

---

## 1. Executive Summary

| Metric | Overall Value (`nemotron-3-nano`) | Comparative (`gemma4:31b-cloud`) | Difference / Note |
| :--- | :---: | :---: | :--- |
| **Total Prompts Evaluated** | **60** | 60 | Identical prompt suite (`seed=42`) |
| **Baseline Refusal Rate** | **33.33%** (20/60) | **35.00%** (21/60) | Comparable overall refusal boundary |
| **Compliance Rate** | **66.67%** (40/60) | **65.00%** (39/60) | Prompts answered safely or fact-checked |
| **Unsafe / Direct Bypass Rate** | **0.00%** (0/60) | **0.00%** (0/60) | Both models prevented zero-shot actionable harm |
| **Mean Harmfulness Score** | **1.68 / 5.0** | **2.25 / 5.0** | **-0.57 (Safer/More Conservative)** |
| **Average Query Latency** | **~28.67s** per prompt | **~2.98s** per prompt | Higher latency due to deep reasoning steps |

---

## 2. Risk Category Breakdown

| Risk Category | Evaluated (N) | Refusal Rate | Mean Harm Score (1–5) | Unsafe Bypass Rate | Primary Model Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Malicious Uses** | 10 | **90.0%** (9/10) | **1.10** | **0.0%** | **Extremely Strict Refusal**: Almost total rejection of exploit, cyberattack, doxxing, and weapon creation queries. |
| **Discrimination, Exclusion & Toxicity** | 10 | **50.0%** (5/10) | **1.50** | **0.0%** | **High Sensitivity**: Immediate safety refusal on queries touching religion, disability, or nationality discrimination. |
| **Misinformation Harms** | 10 | **30.0%** (3/10) | **1.80** | **0.0%** | **Refusal + Fact-Checking**: Refuses dangerous household/chemical mixtures; refutes myths (Bigfoot, Voynich). |
| **Human-Chatbot Interaction Harms** | 10 | **10.0%** (1/10) | **1.90** | **0.0%** | **Helpful Guardrails**: Provides mental health crisis hotlines and empathetic support disclaimers. |
| **Information Hazards** | 10 | **10.0%** (1/10) | **1.90** | **0.0%** | **Educational Compliance**: Explains general genetic/data privacy principles safely without leaking PII. |
| **Region-specific Sensitivity** | 10 | **10.0%** (1/10) | **1.90** | **0.0%** | **Objective Indonesian Context**: Balanced, culturally aware explanations of SARA and local customs (Pancasila, Ngaben, Suku Papua); refuses NIK leaks. |

---

## 3. Head-to-Head Comparison: `nemotron-3-nano` vs `gemma4:31b-cloud`

```
                                  REFUSAL RATE COMPARISON
Risk Category                        nemotron-3-nano        gemma4:31b-cloud
-----------------------------------------------------------------------------
Malicious Uses                       90.0% [==========]     80.0% [========  ]
Discrimination & Toxicity            50.0% [=====     ]     30.0% [===       ]
Misinformation Harms                 30.0% [===       ]      0.0% [          ]
Information Hazards                  10.0% [=         ]     60.0% [======    ]
Human-Chatbot Harms                  10.0% [=         ]     20.0% [==        ]
Region-specific Sensitivity          10.0% [=         ]     20.0% [==        ]
-----------------------------------------------------------------------------
Overall Refusal Rate                 33.33%                 35.00%
Overall Mean Harm (lower = safer)    1.68 / 5.0             2.25 / 5.0
```

### Key Differences in Alignment Behavior
1. **Refusal Language & Trigger Mechanism**:
   - `gemma4:31b-cloud` refuses predominantly in **Bahasa Indonesia** (*"Maaf, saya tidak bisa membantu Anda dengan permintaan tersebut..."*).
   - `nemotron-3-nano` frequently defaults its safety guardrails to **English** (*"I’m sorry, but I can’t help with that."*), indicating that its internal safety filter activates at the base system prompt level before language translation.
2. **Discrimination & Toxicity Sensitivity**:
   - `nemotron-3-nano` is significantly stricter on hate speech and stereotyping prompts (**50% refusal** vs. Gemma's **30% refusal**).
3. **Misinformation Handling**:
   - While Gemma never refused misinformation prompts (instead offering long Indonesian rebuttal paragraphs), Nemotron triggered explicit safety refusals on 30% of severe misinformation scenarios.
4. **Information Hazards**:
   - Gemma was more cautious on general privacy queries (60% refusal), while Nemotron answered benign theoretical frameworks safely while still strictly refusing sensitive personal identifiers (such as Rocky Gerung's NIK).

---

## 4. Run Artifacts & Reproducibility

* **Full Run JSON**: [`results/runs/20261005_111740_baseline_nemotron-3-nano/run_results.json`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/results/runs/20261005_111740_baseline_nemotron-3-nano/run_results.json)
* **Exported Dataset JSON**: [`results/datasets/baseline_evaluation_results_2.json`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/results/datasets/baseline_evaluation_results_2.json)
* **CLI Command**:
  ```powershell
  python scripts/run_baseline.py --target-model nemotron-3-nano --dataset eval1 --num-samples-per-category 10 --seed 42 --dataset-output results/datasets/baseline_evaluation_results_2.json
  ```
