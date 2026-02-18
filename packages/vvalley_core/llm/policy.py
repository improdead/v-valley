"""LLM task policy primitives for modern V-Valley cognition control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ALLOWED_MODEL_TIERS = ("strong", "fast", "cheap", "heuristic")
FALLBACK_CHAIN_BY_TIER: dict[str, tuple[str, ...]] = {
    "strong": ("strong", "fast", "cheap", "heuristic"),
    "fast": ("fast", "cheap", "heuristic"),
    "cheap": ("cheap", "heuristic"),
    "heuristic": ("heuristic",),
}


@dataclass(frozen=True)
class TaskPolicy:
    task_name: str
    model_tier: str
    max_input_tokens: int
    max_output_tokens: int
    temperature: float
    timeout_ms: int
    retry_limit: int
    enable_prompt_cache: bool = True

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["fallback_chain"] = list(resolve_fallback_chain(self.model_tier))
        return out


DEFAULT_TASK_POLICIES: dict[str, TaskPolicy] = {
    "perceive": TaskPolicy(
        task_name="perceive",
        model_tier="fast",
        max_input_tokens=900,
        max_output_tokens=180,
        temperature=0.2,
        timeout_ms=2600,
        retry_limit=1,
    ),
    "retrieve": TaskPolicy(
        task_name="retrieve",
        model_tier="fast",
        max_input_tokens=1200,
        max_output_tokens=220,
        temperature=0.1,
        timeout_ms=2800,
        retry_limit=1,
    ),
    "short_action": TaskPolicy(
        task_name="short_action",
        model_tier="cheap",
        max_input_tokens=280,
        max_output_tokens=48,
        temperature=0.1,
        timeout_ms=1400,
        retry_limit=1,
    ),
    "daily_plan": TaskPolicy(
        task_name="daily_plan",
        model_tier="fast",
        max_input_tokens=1800,
        max_output_tokens=320,
        temperature=0.3,
        timeout_ms=4200,
        retry_limit=2,
    ),
    "long_term_plan": TaskPolicy(
        task_name="long_term_plan",
        model_tier="strong",
        max_input_tokens=3200,
        max_output_tokens=520,
        temperature=0.45,
        timeout_ms=7500,
        retry_limit=2,
    ),
    "plan_move": TaskPolicy(
        task_name="plan_move",
        model_tier="cheap",
        max_input_tokens=280,
        max_output_tokens=48,
        temperature=0.1,
        timeout_ms=1400,
        retry_limit=1,
    ),
    "reflect": TaskPolicy(
        task_name="reflect",
        model_tier="strong",
        max_input_tokens=2200,
        max_output_tokens=320,
        temperature=0.5,
        timeout_ms=6200,
        retry_limit=2,
    ),
    "poignancy": TaskPolicy(
        task_name="poignancy",
        model_tier="cheap",
        max_input_tokens=400,
        max_output_tokens=32,
        temperature=0.0,
        timeout_ms=1200,
        retry_limit=1,
    ),
    "conversation": TaskPolicy(
        task_name="conversation",
        model_tier="fast",
        max_input_tokens=2400,
        max_output_tokens=400,
        temperature=0.6,
        timeout_ms=5000,
        retry_limit=2,
    ),
    "focal_points": TaskPolicy(
        task_name="focal_points",
        model_tier="fast",
        max_input_tokens=1600,
        max_output_tokens=200,
        temperature=0.4,
        timeout_ms=3200,
        retry_limit=1,
    ),
    "reflection_insights": TaskPolicy(
        task_name="reflection_insights",
        model_tier="strong",
        max_input_tokens=2400,
        max_output_tokens=400,
        temperature=0.5,
        timeout_ms=6500,
        retry_limit=2,
    ),
    "schedule_decomp": TaskPolicy(
        task_name="schedule_decomp",
        model_tier="fast",
        max_input_tokens=1200,
        max_output_tokens=320,
        temperature=0.3,
        timeout_ms=3500,
        retry_limit=1,
    ),
    "conversation_summary": TaskPolicy(
        task_name="conversation_summary",
        model_tier="fast",
        max_input_tokens=1400,
        max_output_tokens=240,
        temperature=0.3,
        timeout_ms=3200,
        retry_limit=1,
    ),
    "conversation_turn": TaskPolicy(
        task_name="conversation_turn",
        model_tier="fast",
        max_input_tokens=1800,
        max_output_tokens=120,
        temperature=0.6,
        timeout_ms=2500,
        retry_limit=1,
    ),
    "daily_schedule_gen": TaskPolicy(
        task_name="daily_schedule_gen",
        model_tier="fast",
        max_input_tokens=2400,
        max_output_tokens=600,
        temperature=0.4,
        timeout_ms=5000,
        retry_limit=2,
    ),
    "pronunciatio": TaskPolicy(
        task_name="pronunciatio",
        model_tier="cheap",
        max_input_tokens=300,
        max_output_tokens=48,
        temperature=0.1,
        timeout_ms=1200,
        retry_limit=1,
    ),
    "reaction_policy": TaskPolicy(
        task_name="reaction_policy",
        model_tier="cheap",
        max_input_tokens=800,
        max_output_tokens=64,
        temperature=0.2,
        timeout_ms=1800,
        retry_limit=1,
    ),
    "wake_up_hour": TaskPolicy(
        task_name="wake_up_hour",
        model_tier="cheap",
        max_input_tokens=300,
        max_output_tokens=24,
        temperature=0.1,
        timeout_ms=1200,
        retry_limit=1,
    ),
    "action_address": TaskPolicy(
        task_name="action_address",
        model_tier="cheap",
        max_input_tokens=800,
        max_output_tokens=96,
        temperature=0.1,
        timeout_ms=1800,
        retry_limit=1,
    ),
    "object_state": TaskPolicy(
        task_name="object_state",
        model_tier="cheap",
        max_input_tokens=400,
        max_output_tokens=64,
        temperature=0.1,
        timeout_ms=1400,
        retry_limit=1,
    ),
    "event_reaction": TaskPolicy(
        task_name="event_reaction",
        model_tier="cheap",
        max_input_tokens=600,
        max_output_tokens=64,
        temperature=0.2,
        timeout_ms=1600,
        retry_limit=1,
    ),
    "relationship_summary": TaskPolicy(
        task_name="relationship_summary",
        model_tier="fast",
        max_input_tokens=1200,
        max_output_tokens=160,
        temperature=0.3,
        timeout_ms=2500,
        retry_limit=1,
    ),
    "first_daily_plan": TaskPolicy(
        task_name="first_daily_plan",
        model_tier="fast",
        max_input_tokens=2000,
        max_output_tokens=600,
        temperature=0.5,
        timeout_ms=4500,
        retry_limit=2,
    ),
    "schedule_recomposition": TaskPolicy(
        task_name="schedule_recomposition",
        model_tier="fast",
        max_input_tokens=2000,
        max_output_tokens=600,
        temperature=0.3,
        timeout_ms=4500,
        retry_limit=1,
    ),
    "contextual_prioritize": TaskPolicy(
        task_name="contextual_prioritize",
        model_tier="cheap",
        max_input_tokens=280,
        max_output_tokens=48,
        temperature=0.1,
        timeout_ms=1400,
        retry_limit=1,
    ),
    "identity_evolution": TaskPolicy(
        task_name="identity_evolution",
        model_tier="strong",
        max_input_tokens=2400,
        max_output_tokens=400,
        temperature=0.5,
        timeout_ms=6500,
        retry_limit=2,
    ),
    "social_attitude": TaskPolicy(
        task_name="social_attitude",
        model_tier="fast",
        max_input_tokens=1000,
        max_output_tokens=120,
        temperature=0.3,
        timeout_ms=2600,
        retry_limit=1,
    ),
    "werewolf_night_kill": TaskPolicy(
        task_name="werewolf_night_kill",
        model_tier="cheap",
        max_input_tokens=700,
        max_output_tokens=80,
        temperature=0.2,
        timeout_ms=1800,
        retry_limit=1,
    ),
    "werewolf_doctor_protect": TaskPolicy(
        task_name="werewolf_doctor_protect",
        model_tier="cheap",
        max_input_tokens=700,
        max_output_tokens=80,
        temperature=0.2,
        timeout_ms=1800,
        retry_limit=1,
    ),
    "werewolf_seer_inspect": TaskPolicy(
        task_name="werewolf_seer_inspect",
        model_tier="cheap",
        max_input_tokens=700,
        max_output_tokens=80,
        temperature=0.2,
        timeout_ms=1800,
        retry_limit=1,
    ),
    "werewolf_discussion": TaskPolicy(
        task_name="werewolf_discussion",
        model_tier="fast",
        max_input_tokens=1600,
        max_output_tokens=140,
        temperature=0.5,
        timeout_ms=2600,
        retry_limit=1,
    ),
    "werewolf_vote": TaskPolicy(
        task_name="werewolf_vote",
        model_tier="cheap",
        max_input_tokens=800,
        max_output_tokens=80,
        temperature=0.2,
        timeout_ms=1800,
        retry_limit=1,
    ),
    "anaconda_bet": TaskPolicy(
        task_name="anaconda_bet",
        model_tier="cheap",
        max_input_tokens=700,
        max_output_tokens=90,
        temperature=0.2,
        timeout_ms=1800,
        retry_limit=1,
    ),
    "anaconda_pass": TaskPolicy(
        task_name="anaconda_pass",
        model_tier="cheap",
        max_input_tokens=700,
        max_output_tokens=90,
        temperature=0.1,
        timeout_ms=1800,
        retry_limit=1,
    ),
    "blackjack_bet": TaskPolicy(
        task_name="blackjack_bet",
        model_tier="cheap",
        max_input_tokens=700,
        max_output_tokens=90,
        temperature=0.1,
        timeout_ms=1800,
        retry_limit=1,
    ),
    "blackjack_action": TaskPolicy(
        task_name="blackjack_action",
        model_tier="cheap",
        max_input_tokens=850,
        max_output_tokens=90,
        temperature=0.1,
        timeout_ms=1800,
        retry_limit=1,
    ),
    "holdem_action": TaskPolicy(
        task_name="holdem_action",
        model_tier="cheap",
        max_input_tokens=1100,
        max_output_tokens=100,
        temperature=0.1,
        timeout_ms=2000,
        retry_limit=1,
    ),
}


def normalize_model_tier(value: str) -> str:
    tier = str(value or "").strip().lower()
    if tier not in ALLOWED_MODEL_TIERS:
        raise ValueError(f"Unsupported model tier: {value}")
    return tier


def resolve_fallback_chain(model_tier: str) -> tuple[str, ...]:
    tier = normalize_model_tier(model_tier)
    return FALLBACK_CHAIN_BY_TIER[tier]


def default_policy_for_task(task_name: str) -> TaskPolicy:
    key = str(task_name).strip()
    if key in DEFAULT_TASK_POLICIES:
        return DEFAULT_TASK_POLICIES[key]
    return TaskPolicy(
        task_name=key or "unknown_task",
        model_tier="cheap",
        max_input_tokens=900,
        max_output_tokens=160,
        temperature=0.2,
        timeout_ms=2500,
        retry_limit=1,
    )


def normalize_policy_row(task_name: str, row: dict[str, Any]) -> TaskPolicy:
    return TaskPolicy(
        task_name=task_name,
        model_tier=normalize_model_tier(str(row.get("model_tier") or "cheap")),
        max_input_tokens=max(1, int(row.get("max_input_tokens") or 1)),
        max_output_tokens=max(1, int(row.get("max_output_tokens") or 1)),
        temperature=float(row.get("temperature") if row.get("temperature") is not None else 0.2),
        timeout_ms=max(100, int(row.get("timeout_ms") or 100)),
        retry_limit=max(0, int(row.get("retry_limit") or 0)),
        enable_prompt_cache=bool(row.get("enable_prompt_cache", True)),
    )


def fun_low_cost_preset_policies() -> list[TaskPolicy]:
    return [
        TaskPolicy(
            task_name="perceive",
            model_tier="cheap",
            max_input_tokens=480,
            max_output_tokens=96,
            temperature=0.1,
            timeout_ms=1800,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="retrieve",
            model_tier="cheap",
            max_input_tokens=600,
            max_output_tokens=120,
            temperature=0.1,
            timeout_ms=1800,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="short_action",
            model_tier="cheap",
            max_input_tokens=220,
            max_output_tokens=40,
            temperature=0.1,
            timeout_ms=1200,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="daily_plan",
            model_tier="cheap",
            max_input_tokens=700,
            max_output_tokens=120,
            temperature=0.2,
            timeout_ms=2200,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="long_term_plan",
            model_tier="fast",
            max_input_tokens=1200,
            max_output_tokens=220,
            temperature=0.3,
            timeout_ms=3200,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="plan_move",
            model_tier="cheap",
            max_input_tokens=220,
            max_output_tokens=40,
            temperature=0.1,
            timeout_ms=1200,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="reflect",
            model_tier="fast",
            max_input_tokens=900,
            max_output_tokens=160,
            temperature=0.3,
            timeout_ms=2800,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="poignancy",
            model_tier="cheap",
            max_input_tokens=300,
            max_output_tokens=24,
            temperature=0.0,
            timeout_ms=1000,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="conversation",
            model_tier="cheap",
            max_input_tokens=900,
            max_output_tokens=200,
            temperature=0.5,
            timeout_ms=2800,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="focal_points",
            model_tier="cheap",
            max_input_tokens=700,
            max_output_tokens=120,
            temperature=0.3,
            timeout_ms=2000,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="reflection_insights",
            model_tier="fast",
            max_input_tokens=1000,
            max_output_tokens=200,
            temperature=0.4,
            timeout_ms=3200,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="schedule_decomp",
            model_tier="cheap",
            max_input_tokens=600,
            max_output_tokens=160,
            temperature=0.2,
            timeout_ms=2000,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="conversation_summary",
            model_tier="cheap",
            max_input_tokens=700,
            max_output_tokens=120,
            temperature=0.2,
            timeout_ms=1800,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="conversation_turn",
            model_tier="cheap",
            max_input_tokens=900,
            max_output_tokens=80,
            temperature=0.5,
            timeout_ms=1800,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="daily_schedule_gen",
            model_tier="cheap",
            max_input_tokens=1200,
            max_output_tokens=400,
            temperature=0.3,
            timeout_ms=3500,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="pronunciatio",
            model_tier="cheap",
            max_input_tokens=200,
            max_output_tokens=32,
            temperature=0.1,
            timeout_ms=1000,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="reaction_policy",
            model_tier="cheap",
            max_input_tokens=500,
            max_output_tokens=48,
            temperature=0.1,
            timeout_ms=1400,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="wake_up_hour",
            model_tier="cheap",
            max_input_tokens=200,
            max_output_tokens=16,
            temperature=0.0,
            timeout_ms=1000,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="action_address",
            model_tier="cheap",
            max_input_tokens=500,
            max_output_tokens=64,
            temperature=0.1,
            timeout_ms=1400,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="object_state",
            model_tier="cheap",
            max_input_tokens=300,
            max_output_tokens=48,
            temperature=0.1,
            timeout_ms=1200,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="event_reaction",
            model_tier="cheap",
            max_input_tokens=400,
            max_output_tokens=48,
            temperature=0.1,
            timeout_ms=1200,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="relationship_summary",
            model_tier="cheap",
            max_input_tokens=600,
            max_output_tokens=96,
            temperature=0.2,
            timeout_ms=1800,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="first_daily_plan",
            model_tier="cheap",
            max_input_tokens=1000,
            max_output_tokens=400,
            temperature=0.3,
            timeout_ms=3000,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="schedule_recomposition",
            model_tier="cheap",
            max_input_tokens=1000,
            max_output_tokens=400,
            temperature=0.2,
            timeout_ms=3000,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="blackjack_bet",
            model_tier="cheap",
            max_input_tokens=600,
            max_output_tokens=80,
            temperature=0.1,
            timeout_ms=1600,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="blackjack_action",
            model_tier="cheap",
            max_input_tokens=700,
            max_output_tokens=80,
            temperature=0.1,
            timeout_ms=1600,
            retry_limit=1,
        ),
        TaskPolicy(
            task_name="holdem_action",
            model_tier="cheap",
            max_input_tokens=800,
            max_output_tokens=80,
            temperature=0.1,
            timeout_ms=1700,
            retry_limit=1,
        ),
    ]


def situational_default_preset_policies() -> list[TaskPolicy]:
    # Default preset: large budgets for long-term/daily planning, small for short actions.
    return [DEFAULT_TASK_POLICIES[k] for k in sorted(DEFAULT_TASK_POLICIES.keys())]


def estimate_token_count(text: str) -> int:
    # Cheap estimate that keeps routing deterministic and provider-agnostic.
    return max(1, len(text) // 4)


def trim_context_to_budget(context_text: str, max_input_tokens: int) -> tuple[str, bool, int]:
    estimated = estimate_token_count(context_text)
    if estimated <= max_input_tokens:
        return context_text, False, estimated

    max_chars = max(32, int(max_input_tokens * 4))
    trimmed = context_text[-max_chars:]
    return trimmed, True, estimate_token_count(trimmed)
