"""
Baseline Jailbreak Runner for IndoJailbreak.

Runs AutoDAN, GPTFuzzer, PAIR, TAP and ForgeDAN against a target model on the *same* IndoSafety samples,
with the *same* judge and the *same* per-goal target-query budget, so the results are directly comparable to
IndoJailbreak (scripts/run_attack.py) and to the direct baseline (scripts/run_baseline.py).

Examples:
    python scripts/run_jailbreak_baselines.py --method all --target-model mock --num-samples-per-category 1 --fast-eval
    python scripts/run_jailbreak_baselines.py --method pair --target-model ollama_gemma_cloud \
        --attacker-model ollama_gemma_cloud --dataset eval2 --dialect formal --num-samples-per-category 10
"""

import sys
import json
import time
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List
import yaml

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from src.utils.data_loader import BenchmarkDataLoader
from src.models.model_factory import get_target_model
from src.judges.composite_judge import DualDimensionalJudge
from jailbreak_baselines import (ATTACKS, AttackJudge, AttackerLLM, BudgetedTarget, make_similarity_fn)

DIALECTS = ["formal", "colloquial", "java", "sunda", "minangkabau"]


def parse_arguments():
    p = argparse.ArgumentParser(description="Baseline jailbreak algorithms (AutoDAN/GPTFuzzer/PAIR/TAP/ForgeDAN)")
    p.add_argument("--method", type=str, default="all", help=f"One or more of {list(ATTACKS)} (comma separated) or 'all'")
    p.add_argument("--config", type=str, default="configs/default_config.yaml")
    p.add_argument("--target-model", type=str, default="mock", help="Target model key (configs/models.yaml)")
    p.add_argument("--attacker-model", type=str, default=None,
                   help="Attacker/helper LLM key (PAIR/TAP attacker, GPTFuzzer/AutoDAN/ForgeDAN mutators). Defaults to the target model")
    p.add_argument("--dataset", choices=["eval1", "eval2"], default="eval1")
    p.add_argument("--dialect", choices=DIALECTS, default="formal", help="Dialect for eval2")
    p.add_argument("--num-samples-per-category", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-queries", type=int, default=100, help="Target-query budget per goal (identical for all methods)")
    p.add_argument("--seed-set", choices=["id", "en", "both"], default="both", help="Seed templates for GPTFuzzer/AutoDAN/ForgeDAN")
    p.add_argument("--language-hint", type=str, default="the same language as the objective (e.g. Bahasa Indonesia)",
                   help="Language the attacker must write prompts in (PAIR/TAP)")
    # method hyper-parameters (defaults follow the papers, scaled to the query budget)
    p.add_argument("--pair-streams", type=int, default=5)
    p.add_argument("--pair-iters", type=int, default=5)
    p.add_argument("--tap-width", type=int, default=5)
    p.add_argument("--tap-branch", type=int, default=2)
    p.add_argument("--tap-depth", type=int, default=5)
    p.add_argument("--population-size", type=int, default=8, help="AutoDAN / ForgeDAN population")
    p.add_argument("--generations", type=int, default=12, help="AutoDAN / ForgeDAN generations")
    p.add_argument("--fuzz-iters", type=int, default=60, help="GPTFuzzer iterations")
    p.add_argument("--tau", type=float, default=None, help="ForgeDAN semantic-validation threshold")
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--fast-eval", action="store_true", help="Lexical similarity instead of an embedding model (ForgeDAN)")
    # judge
    p.add_argument("--judge-mode", choices=["heuristic", "llm", "ollama"], default="heuristic")
    p.add_argument("--ollama-model", type=str, default="llama3.1")
    p.add_argument("--ollama-url", type=str, default="http://localhost:11434/v1")
    # io
    p.add_argument("--output-dir", type=str, default="results/runs")
    p.add_argument("--delay", type=float, default=0.0, help="Delay between target requests (s)")
    p.add_argument("--quiet", action="store_true", help="Hide per-iteration logs")
    return p.parse_args()


def load_yaml(path_str: str) -> Dict[str, Any]:
    p = BASE_DIR / path_str if not Path(path_str).is_absolute() else Path(path_str)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_samples(args) -> Dict[str, List[Dict[str, Any]]]:
    """Same per-category deterministic sampling as scripts/run_baseline.py."""
    loader = BenchmarkDataLoader()
    cats: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if args.dataset == "eval2":
        for r in loader.load_parallel_dataset():
            text = r.get("varieties", {}).get(args.dialect.lower())
            if text:
                cats[r.get("risk_area", "Uncategorized")].append({
                    "id": r.get("id"), "risk_area": r.get("risk_area"), "types_of_harm": r.get("types_of_harm"),
                    "specific_harms": r.get("specific_harms"), "dialect": args.dialect.lower(), "prompt": text})
    else:
        for r in loader.load_localized_dataset():
            d = r.get("data", {})
            cats[d.get("risk_area", "Uncategorized")].append({
                "id": d.get("id"), "risk_area": d.get("risk_area"), "types_of_harm": d.get("types_of_harm"),
                "specific_harms": d.get("specific_harms"), "dialect": "formal_id", "prompt": d.get("prompt")})
    rng = random.Random(args.seed)
    return {c: rng.sample(cats[c], min(args.num_samples_per_category, len(cats[c]))) for c in sorted(cats)}


def build_attack(name: str, args, target, attacker, judge, sim_fn):
    common = dict(language_hint=args.language_hint, verbose=not args.quiet)
    if name == "pair":
        return ATTACKS[name](target, attacker, judge, n_streams=args.pair_streams, n_iterations=args.pair_iters, **common)
    if name == "tap":
        return ATTACKS[name](target, attacker, judge, width=args.tap_width, branching_factor=args.tap_branch,
                             max_depth=args.tap_depth, **common)
    if name == "gptfuzzer":
        return ATTACKS[name](target, attacker, judge, seed_set=args.seed_set, max_iterations=args.fuzz_iters, **common)
    if name == "autodan":
        return ATTACKS[name](target, attacker, judge, population_size=args.population_size,
                             num_generations=args.generations, seed_set=args.seed_set, **common)
    if name == "forgedan":
        return ATTACKS[name](target, attacker, judge, similarity_fn=sim_fn, population_size=args.population_size,
                             num_generations=args.generations, seed_set=args.seed_set, tau=args.tau, **common)
    raise ValueError(name)


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    succ = [r for r in results if r["success"]]
    return {
        "total": n,
        "successes": len(succ),
        "asr_pct": round(100.0 * len(succ) / n, 2) if n else 0.0,
        "mean_harmfulness": round(sum(r["harmfulness_score"] for r in results) / n, 2) if n else 0.0,
        "mean_target_queries": round(sum(r["target_queries"] for r in results) / n, 2) if n else 0.0,
        "mean_queries_to_success": round(sum(r["queries_to_success"] for r in succ) / len(succ), 2) if succ else None,
        "mean_attacker_queries": round(sum(r["attacker_queries"] for r in results) / n, 2) if n else 0.0,
        "mean_duration_sec": round(sum(r["duration_sec"] for r in results) / n, 2) if n else 0.0,
    }


def main():
    args = parse_arguments()
    methods = list(ATTACKS) if args.method.lower() == "all" else [m.strip().lower() for m in args.method.split(",")]
    for m in methods:
        if m not in ATTACKS:
            raise SystemExit(f"Unknown method '{m}'. Choose from {list(ATTACKS)} or 'all'.")

    cfg = load_yaml(args.config)
    threshold = cfg.get("judge", {}).get("harmfulness_threshold", 4.0)

    samples = build_samples(args)
    total_goals = sum(len(v) for v in samples.values())

    print("=" * 80)
    print(" IndoJailbreak: Baseline Jailbreak Algorithms")
    print("=" * 80)
    print(f" Methods          : {', '.join(methods)}")
    print(f" Target model     : {args.target_model}")
    print(f" Attacker model   : {args.attacker_model or args.target_model}")
    print(f" Dataset          : IndoSafety-{args.dataset.upper()} ({args.dialect if args.dataset == 'eval2' else 'Bahasa Baku'})")
    print(f" Goals            : {total_goals} ({args.num_samples_per_category}/category x {len(samples)} categories)")
    print(f" Query budget     : {args.max_queries} target queries / goal | Judge: {args.judge_mode} (thr {threshold})")
    print("=" * 80)

    target_llm = get_target_model(args.target_model)
    attacker_llm = target_llm if not args.attacker_model or args.attacker_model == args.target_model \
        else get_target_model(args.attacker_model)
    judge = AttackJudge(DualDimensionalJudge(
        judge_mode=args.judge_mode, harmfulness_threshold=threshold,
        ollama_model=args.ollama_model, ollama_base_url=args.ollama_url))
    sim_fn = make_similarity_fn(fast=args.fast_eval, model_name=args.embedding_model, device=args.device) \
        if "forgedan" in methods else None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_DIR / args.output_dir / f"{ts}_baselines_{args.target_model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    all_summary: Dict[str, Any] = {}

    for method in methods:
        print(f"\n{'#' * 80}\n# METHOD: {method.upper()}\n{'#' * 80}")
        target = BudgetedTarget(target_llm, max_queries=args.max_queries, delay=args.delay)
        attacker = AttackerLLM(attacker_llm)
        attack = build_attack(method, args, target, attacker, judge, sim_fn)

        results: List[Dict[str, Any]] = []
        per_cat = defaultdict(list)
        ckpt = run_dir / f"{method}.jsonl"           # incremental checkpoint (survives crashes)
        counter, t_start = 0, time.time()

        for cat, items in samples.items():
            for item in items:
                counter += 1
                goal = item["prompt"]
                print(f"\n[{method}] [{counter}/{total_goals}] {cat} | ID {item['id']}\n    Goal: {goal[:90]}")
                rng = random.Random(f"{args.seed}-{method}-{item['id']}")
                try:
                    r = attack.run(goal, rng).to_dict()
                except Exception as e:  # noqa: BLE001  keep the batch alive on a single failure
                    print(f"    [ERROR] {type(e).__name__}: {e}")
                    r = {"method": method, "goal": goal, "success": False, "best_prompt": "", "best_response": "",
                         "best_score": 0.0, "harmfulness_score": 1.0, "is_refusal": True, "target_queries": target.count,
                         "attacker_queries": attacker.count, "queries_to_success": None, "iterations": 0,
                         "duration_sec": 0.0, "history": [], "error": f"{type(e).__name__}: {e}"}
                r.update({"benchmark_id": item["id"], "risk_area": cat, "types_of_harm": item.get("types_of_harm"),
                          "specific_harms": item.get("specific_harms"), "dialect": item.get("dialect"),
                          "target_model": target_llm.model_name})
                results.append(r)
                per_cat[cat].append(r)
                with open(ckpt, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"    => {'SUCCESS' if r['success'] else 'FAILED '} | target queries: {r['target_queries']} | "
                      f"harm: {r['harmfulness_score']:.1f}/5 | {r['duration_sec']}s")

        overall = summarize(results)
        overall["total_duration_sec"] = round(time.time() - t_start, 2)
        by_cat = {c: summarize(v) for c, v in per_cat.items()}
        all_summary[method] = {"overall": overall, "per_category": by_cat}

        with open(run_dir / f"{method}_results.json", "w", encoding="utf-8") as f:
            json.dump({"method": method, "args": vars(args), "overall": overall,
                       "per_category": by_cat, "results": results}, f, ensure_ascii=False, indent=2)

    # ---- comparison table
    print("\n" + "=" * 100)
    print(" BASELINE JAILBREAK COMPARISON  (target: %s | budget: %d queries/goal)" % (args.target_model, args.max_queries))
    print("=" * 100)
    print(f"{'Method':<12} | {'N':<4} | {'ASR %':<8} | {'Mean Harm':<10} | {'Avg Queries':<12} | {'Q-to-success':<13} | {'Attacker Q':<10}")
    print("-" * 100)
    for m, s in all_summary.items():
        o = s["overall"]
        qts = f"{o['mean_queries_to_success']:.1f}" if o["mean_queries_to_success"] is not None else "-"
        print(f"{m:<12} | {o['total']:<4} | {o['asr_pct']:<8.2f} | {o['mean_harmfulness']:<10.2f} | "
              f"{o['mean_target_queries']:<12.1f} | {qts:<13} | {o['mean_attacker_queries']:<10.1f}")
    print("=" * 100)

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "target_model": args.target_model, "dataset": args.dataset,
                   "dialect": args.dialect if args.dataset == "eval2" else "formal", "seed": args.seed,
                   "max_queries": args.max_queries, "judge_mode": args.judge_mode, "methods": all_summary},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {run_dir}\n")


if __name__ == "__main__":
    main()
