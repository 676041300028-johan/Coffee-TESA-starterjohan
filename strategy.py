"""Main strategy entry point delegating to RL adapter and rule fallbacks."""
import random
from typing import Any

from rl_adapter import run_rl_agent
from fallback_rules import candidates, rule_plan, guard  # noqa: F401 — exported for main.py


class Context:
    """Context object instantiated by main.py every round."""
    def __init__(
        self,
        rules: dict,
        calendar: Any,
        tv: dict[str, float],
        rng: random.Random,
        memory: Any,
        llm: Any = None,
        seconds_left: float = 999.0,
    ):
        self.rules = rules
        self.calendar = calendar
        self.tv = tv
        self.rng = rng
        self.memory = memory
        self.llm = llm
        self.seconds_left = seconds_left
        self.notes: list[str] = []


def decide(state: dict, ctx: Context) -> tuple[list[dict], str]:
    try:
        rl_offers, tag = run_rl_agent(state, ctx)
        safe_offers = guard(state, ctx, rl_offers)
        if safe_offers:
            return safe_offers, tag
    except Exception as e:
        print(f"❌ RL Error: {e}", flush=True)
        ctx.notes.append(f"RL error: {e}")

    return guard(state, ctx, rule_plan(state, ctx, candidates(state, ctx))), "rule_fallback"