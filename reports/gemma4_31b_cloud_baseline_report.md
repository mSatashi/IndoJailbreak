# IndoSafety Baseline Safety Evaluation Report: `ollama_gemma_cloud`

**Date:** 2026-10-05  
**Model Under Test:** `ollama_gemma_cloud` (`gemma4:31b-cloud`)  
**Benchmark:** [IndoSafety-Eval-1](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/data/localized/indosafety_eval1_all.json) (Bahasa Indonesia Baku)  
**Sample Strategy:** 10 deterministically sampled prompts per risk category (`seed=42`, total 60 prompts)  
**Evaluation Mode:** Direct Baseline Evaluation (Dual-Dimensional Judge: Refusal Detection + Actionable Harmfulness Scoring)  

---

## 1. Executive Summary

| Metric | Overall Value | Description |
| :--- | :--- | :--- |
| **Total Prompts Evaluated** | **60** | 10 prompts across 6 risk domains |
| **Baseline Refusal Rate** | **35.00%** (21/60) | Explicit safety refusals (*"Maaf, saya tidak bisa..."*) |
| **Compliance Rate** | **65.00%** (39/60) | Prompts answered or fact-checked/debunked |
| **Unsafe / Actionable Bypass Rate** | **0.00%** (0/60) | Zero direct/zero-shot jailbreak vulnerabilities found |
| **Mean Harmfulness Score** | **2.25 / 5.0** | Low-to-moderate risk profile |
| **Evaluation Duration** | **179.09s** (~2.98 mins) | Mean latency: ~2.98s per query |

---

## 2. Risk Category Breakdown

The benchmark categorizes prompts into 6 distinct harm areas:

| Risk Category | Evaluated (N) | Refusal Rate | Mean Harm Score (1-5) | Unsafe Bypass Rate | Primary Model Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Malicious Uses** | 10 | **80.0%** (8/10) | **1.40** | **0.0%** | **Strong Refusal**: High sensitivity to exploit/attack tools, social engineering, and doxxing. |
| **Information Hazards** | 10 | **60.0%** (6/10) | **2.10** | **0.0%** | **Moderate-High Refusal**: Refuses requests for PII, biological sequences, or sensitive credentials. |
| **Discrimination, Exclusion & Toxicity** | 10 | **30.0%** (3/10) | **2.30** | **0.0%** | **Contextual / Nuanced**: Refuses overt slurs/hate speech; provides neutral sociolinguistic context on sensitive queries. |
| **Region-specific Sensitivity** | 10 | **20.0%** (2/10) | **2.40** | **0.0%** | **Neutral Fact-Checking**: Rebuts stereotypes (e.g. SARA/regional stereotypes) with balanced explanations; refuses PII (e.g., Rocky Gerung's NIK). |
| **Human-Chatbot Interaction Harms** | 10 | **20.0%** (2/10) | **2.30** | **0.0%** | **Safety Disclaimers**: Disclaims AI limits (*"Saya bukan dokter"*), refuses extreme self-harm, offers harm-reduction advice. |
| **Misinformation Harms** | 10 | **0.0%** (0/10) | **3.00** | **0.0%** | **Proactive Debunking**: Rather than outright refusing, the model actively debunks dangerous advice (e.g., pouring water on boiling oil, jumping jacks after snakebite) with bold warnings. |

---

## 3. Key Observations & Alignment Analysis

### Strong Areas
1. **Malicious Uses (80% Refusal Rate)**:  
   The model exhibits strong pre-trained safety boundaries against requests for doxxing, exploiting security flaws, or creating harmful scripts.
2. **Actionable Harm Prevention (0% Unsafe Bypass)**:  
   Across all 60 baseline queries, the model did not generate actionable harmful guides. Whenever it complied with answering a sensitive prompt, it provided educational, historical, or counter-misinformation context.
3. **Medical & Disinformation Fact-Checking**:  
   In *Misinformation Harms*, the model systematically added `**PERINGATAN KERAS: JANGAN PERNAH melakukan hal ini**` and scientifically explained why erroneous claims (e.g., swallowing soda for dental plaque, snakebite physical exertion) are medically hazardous.

### Vulnerabilities / Areas for Evolutionary Adversarial Red-Teaming
* **Misinformation & Region-Specific SARA Inquiries**:  
   Because the model prefers to engage in dialectical explanations rather than hard refusal when queries are framed theoretically or historically, it represents prime targets for evolutionary framing mutations (e.g. AutoDAN / ForgeDAN character leetspeak, dialect transfer to *colloquial* or *Basa Jawa/Sunda*).

---

## 4. Run Artifacts & Reproducibility

* **Full Run JSON**: [`results/runs/20261005_105111_baseline_ollama_gemma_cloud/run_results.json`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/results/runs/20261005_105111_baseline_ollama_gemma_cloud/run_results.json)
* **Exported Dataset JSON**: [`results/datasets/baseline_evaluation_results.json`](file:///c:/Users/SAYID/DiyyasIserafna/tesis_prep/Tesis_Project/results/datasets/baseline_evaluation_results.json)
* **CLI Command**:
  ```powershell
  python scripts/run_baseline.py --target-model ollama_gemma_cloud --dataset eval1 --num-samples-per-category 10 --seed 42
  ```
