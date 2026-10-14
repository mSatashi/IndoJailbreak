"""Baseline jailbreak attacks for comparison against IndoJailbreak."""

from .common import (Attack, AttackResult, AttackJudge, AttackerLLM, BudgetedTarget, BudgetExhausted,
                     make_similarity_fn)
from .autodan import AutoDANAttack
from .gptfuzzer import GPTFuzzerAttack
from .pair import PAIRAttack
from .tap import TAPAttack
from .forgedan import ForgeDANAttack

ATTACKS = {
    "autodan": AutoDANAttack,
    "gptfuzzer": GPTFuzzerAttack,
    "pair": PAIRAttack,
    "tap": TAPAttack,
    "forgedan": ForgeDANAttack,
}

__all__ = ["ATTACKS", "Attack", "AttackResult", "AttackJudge", "AttackerLLM", "BudgetedTarget",
           "BudgetExhausted", "make_similarity_fn", "AutoDANAttack", "GPTFuzzerAttack", "PAIRAttack",
           "TAPAttack", "ForgeDANAttack"]
