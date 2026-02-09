"""Provider adapters for policy-tiered cognition execution.

This module uses an OpenAI-compatible Chat Completions API contract so a
single integration path can work across multiple model vendors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib import error, request
import json
import os

from .policy import estimate_token_count


DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://api.openai.com/v1"
SUPPORTED_PROVIDERS = {"openai_compatible"}
DEFAULT_MODEL_BY_TIER: dict[str, str] = {
    "strong": "gpt-5.2",
    "fast": "gpt-5-mini",
    "cheap": "gpt-5-nano",
}

_JSON_SCHEMA_TASKS: dict[str, dict[str, Any]] = {
    "plan_move": {
        "type": "object",
        "properties": {
            "dx": {"type": "integer", "minimum": -1, "maximum": 1},
            "dy": {"type": "integer", "minimum": -1, "maximum": 1},
            "reason": {"type": "string", "maxLength": 200},
            "target_x": {"type": "integer"},
            "target_y": {"type": "integer"},
            "goal_reason": {"type": "string", "maxLength": 240},
            "affordance": {"type": "string", "maxLength": 120},
        },
        "required": ["dx", "dy", "reason"],
        "additionalProperties": False,
    },
    "short_action": {
        "type": "object",
        "properties": {
            "dx": {"type": "integer", "minimum": -1, "maximum": 1},
            "dy": {"type": "integer", "minimum": -1, "maximum": 1},
            "reason": {"type": "string", "maxLength": 200},
        },
        "required": ["dx", "dy", "reason"],
        "additionalProperties": False,
    },
    "reaction_policy": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["talk", "wait", "ignore"]},
            "reason": {"type": "string", "maxLength": 240},
            "wait_steps": {"type": "integer", "minimum": 0, "maximum": 8},
        },
        "required": ["decision", "reason"],
        "additionalProperties": False,
    },
    "poignancy": {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["score"],
        "additionalProperties": False,
    },
    "conversation": {
        "type": "object",
        "properties": {
            "utterances": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string"},
                        "text": {"type": "string", "maxLength": 300},
                    },
                    "required": ["speaker", "text"],
                    "additionalProperties": False,
                },
            },
            "summary": {"type": "string", "maxLength": 400},
        },
        "required": ["utterances", "summary"],
        "additionalProperties": False,
    },
    "focal_points": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {"type": "string", "maxLength": 200},
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    },
    "reflection_insights": {
        "type": "object",
        "properties": {
            "insights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "insight": {"type": "string", "maxLength": 300},
                        "evidence_indices": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                        },
                    },
                    "required": ["insight", "evidence_indices"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["insights"],
        "additionalProperties": False,
    },
    "schedule_decomp": {
        "type": "object",
        "properties": {
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "maxLength": 200},
                        "duration_mins": {"type": "integer", "minimum": 1},
                    },
                    "required": ["description", "duration_mins"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["subtasks"],
        "additionalProperties": False,
    },
    "conversation_summary": {
        "type": "object",
        "properties": {
            "memo": {"type": "string", "maxLength": 400},
            "planning_thought": {"type": "string", "maxLength": 400},
        },
        "required": ["memo", "planning_thought"],
        "additionalProperties": False,
    },
    "conversation_turn": {
        "type": "object",
        "properties": {
            "utterance": {"type": "string", "maxLength": 300},
            "end_conversation": {"type": "boolean"},
        },
        "required": ["utterance", "end_conversation"],
        "additionalProperties": False,
    },
    "daily_schedule_gen": {
        "type": "object",
        "properties": {
            "schedule": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "maxLength": 200},
                        "duration_mins": {"type": "integer", "minimum": 1},
                    },
                    "required": ["description", "duration_mins"],
                    "additionalProperties": False,
                },
            },
            "currently": {"type": "string", "maxLength": 300},
        },
        "required": ["schedule", "currently"],
        "additionalProperties": False,
    },
    "pronunciatio": {
        "type": "object",
        "properties": {
            "emoji": {"type": "string", "maxLength": 10},
            "event_triple": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "maxLength": 100},
                    "predicate": {"type": "string", "maxLength": 100},
                    "object": {"type": "string", "maxLength": 100},
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            },
        },
        "required": ["emoji", "event_triple"],
        "additionalProperties": False,
    },
    "wake_up_hour": {
        "type": "object",
        "properties": {
            "wake_up_hour": {"type": "integer", "minimum": 0, "maximum": 23},
        },
        "required": ["wake_up_hour"],
        "additionalProperties": False,
    },
    "action_address": {
        "type": "object",
        "properties": {
            "action_sector": {"type": "string"},
            "action_arena": {"type": "string"},
            "action_game_object": {"type": "string"},
        },
        "required": ["action_sector"],
        "additionalProperties": False,
    },
    "object_state": {
        "type": "object",
        "properties": {
            "object_description": {"type": "string"},
            "object_event_triple": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
            },
        },
        "required": ["object_description"],
        "additionalProperties": False,
    },
    "event_reaction": {
        "type": "object",
        "properties": {
            "react": {"type": "boolean"},
            "reason": {"type": "string"},
            "new_action": {"type": "string"},
        },
        "required": ["react", "reason"],
        "additionalProperties": False,
    },
    "relationship_summary": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "conversation_tone": {"type": "string"},
        },
        "required": ["summary"],
        "additionalProperties": False,
    },
    "first_daily_plan": {
        "type": "object",
        "properties": {
            "schedule": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "duration_mins": {"type": "integer"},
                    },
                },
            },
            "currently": {"type": "string"},
        },
        "required": ["schedule", "currently"],
        "additionalProperties": False,
    },
    "schedule_recomposition": {
        "type": "object",
        "properties": {
            "schedule": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "duration_mins": {"type": "integer"},
                    },
                },
            },
            "inserted_activity": {"type": "string"},
            "inserted_duration_mins": {"type": "integer"},
        },
        "required": ["schedule"],
        "additionalProperties": False,
    },
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _schema_path(parent: str, key: str) -> str:
    if not parent:
        return key
    if key.startswith("["):
        return f"{parent}{key}"
    return f"{parent}.{key}"


def _validate_value_against_schema(*, value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    expected = schema.get("type")

    if expected == "object":
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object")
            return
        properties = schema.get("properties")
        property_schemas = properties if isinstance(properties, dict) else {}
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                field = str(key)
                if field not in value:
                    errors.append(f"{path}: missing required property '{field}'")
        for key, prop_schema in property_schemas.items():
            if key not in value or not isinstance(prop_schema, dict):
                continue
            _validate_value_against_schema(
                value=value[key],
                schema=prop_schema,
                path=_schema_path(path, str(key)),
                errors=errors,
            )
        additional = schema.get("additionalProperties", True)
        if additional is False:
            for key in value.keys():
                if key not in property_schemas:
                    errors.append(f"{path}: unexpected property '{key}'")
        elif isinstance(additional, dict):
            for key, child in value.items():
                if key in property_schemas:
                    continue
                _validate_value_against_schema(
                    value=child,
                    schema=additional,
                    path=_schema_path(path, str(key)),
                    errors=errors,
                )
        return

    if expected == "array":
        if not isinstance(value, list):
            errors.append(f"{path}: expected array")
            return
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_value_against_schema(
                    value=item,
                    schema=item_schema,
                    path=_schema_path(path, f"[{index}]"),
                    errors=errors,
                )
        return

    if expected == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string")
            return
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            errors.append(f"{path}: length < {min_length}")
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and len(value) > max_length:
            errors.append(f"{path}: length > {max_length}")
    elif expected == "integer":
        if not _is_int(value):
            errors.append(f"{path}: expected integer")
            return
    elif expected == "number":
        if not _is_number(value):
            errors.append(f"{path}: expected number")
            return
    elif expected == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: expected boolean")
            return

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values and value not in enum_values:
        errors.append(f"{path}: value not in enum")

    if _is_number(value):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path}: value < {minimum}")
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path}: value > {maximum}")


def _validate_output_schema(task_name: str, output: dict[str, Any]) -> None:
    schema = _schema_for_task(task_name)
    if not schema:
        return
    errors: list[str] = []
    _validate_value_against_schema(value=output, schema=schema, path="$", errors=errors)
    if errors:
        preview = "; ".join(errors[:3])
        raise ProviderExecutionError(
            f"Output schema validation failed for task '{task_name}': {preview}",
            error_code="invalid_schema_output",
        )


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def _schema_for_task(task_name: str) -> dict[str, Any] | None:
    key = str(task_name or "").strip().lower()
    if key in _JSON_SCHEMA_TASKS:
        return _JSON_SCHEMA_TASKS[key]
    if key in {"daily_plan", "long_term_plan"}:
        return {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "maxLength": 240},
            },
            "required": ["reason"],
            "additionalProperties": False,
        }
    return None


def _response_format_for_task(task_name: str) -> dict[str, Any] | None:
    if not _truthy_env("VVALLEY_LLM_USE_JSON_SCHEMA", True):
        return None
    schema = _schema_for_task(task_name)
    if not schema:
        return None
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in str(task_name or "task"))[:48] or "task"
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"vvalley_{safe_name}",
            "strict": True,
            "schema": schema,
        },
    }


@dataclass(frozen=True)
class ProviderExecutionResult:
    output: dict[str, Any]
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, error_code: str, model_name: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.model_name = model_name


class ProviderUnavailableError(ProviderError):
    pass


class ProviderExecutionError(ProviderError):
    pass


@dataclass(frozen=True)
class TierProviderConfig:
    tier: str
    provider: str
    model: str
    base_url: str
    api_key: str | None

    def model_name(self) -> str:
        return f"{self.provider}:{self.model}"


def default_model_for_tier(tier: str) -> str | None:
    tier_name = str(tier or "").strip().lower()
    return DEFAULT_MODEL_BY_TIER.get(tier_name)


def _tier_provider_config(tier: str) -> TierProviderConfig:
    tier_name = str(tier or "").strip().lower()
    if not tier_name:
        raise ProviderUnavailableError("Missing tier name", error_code="missing_tier")

    env_tier = tier_name.upper()
    provider = (
        _first_non_empty(
            os.environ.get(f"VVALLEY_LLM_{env_tier}_PROVIDER"),
            os.environ.get("VVALLEY_LLM_PROVIDER"),
        )
        or "openai_compatible"
    ).lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ProviderUnavailableError(
            f"Unsupported provider: {provider}",
            error_code="unsupported_provider",
        )

    model = _first_non_empty(
        os.environ.get(f"VVALLEY_LLM_{env_tier}_MODEL"),
        os.environ.get("VVALLEY_LLM_MODEL"),
        default_model_for_tier(tier_name),
    )
    if not model:
        raise ProviderUnavailableError(
            f"No model configured for tier: {tier_name}",
            error_code="missing_model",
        )

    base_url = _first_non_empty(
        os.environ.get(f"VVALLEY_LLM_{env_tier}_BASE_URL"),
        os.environ.get("VVALLEY_LLM_BASE_URL"),
    )
    if not base_url:
        base_url = DEFAULT_OPENAI_COMPATIBLE_BASE_URL

    api_key = _first_non_empty(
        os.environ.get(f"VVALLEY_LLM_{env_tier}_API_KEY"),
        os.environ.get("VVALLEY_LLM_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
    )
    if not api_key and not _truthy_env("VVALLEY_LLM_ALLOW_EMPTY_API_KEY", False):
        raise ProviderUnavailableError(
            f"No API key configured for tier: {tier_name}",
            error_code="missing_api_key",
        )

    return TierProviderConfig(
        tier=tier_name,
        provider=provider,
        model=model,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
    )


_TASK_PROMPTS: dict[str, tuple[str, str]] = {
    "plan_move": (
        "You are a V-Valley agent movement planner. The agent has a persona with personality traits, "
        "occupation, and goals — use these to inform movement decisions. Return a JSON object with integer dx and dy "
        "(each -1, 0, or 1), a short reason string, and optionally target_x, target_y, goal_reason, affordance.",
        "Decide the agent's next movement step based on their persona and context below.\n"
        "Output: JSON with dx, dy (integers -1..1), reason (string).\n\n"
    ),
    "short_action": (
        "You are a V-Valley agent action planner. Return a JSON object with integer dx and dy "
        "(each -1, 0, or 1) and a short reason string.",
        "Decide the agent's next short action step.\n"
        "Output: JSON with dx, dy (integers -1..1), reason (string).\n\n"
    ),
    "reaction_policy": (
        "You are a V-Valley social reaction evaluator. Each agent has a persona with personality traits "
        "and background — use these to decide whether they should talk, wait, or ignore another nearby agent. "
        "Return a JSON object.",
        "Evaluate whether the agent should initiate conversation, considering their persona.\n"
        "Output: JSON with decision ('talk', 'wait', or 'ignore'), reason (string), "
        "and optionally wait_steps (integer 0-8).\n\n"
    ),
    "poignancy": (
        "You are a V-Valley event importance scorer. Rate the importance/poignancy of an event "
        "on a scale of 1 (mundane) to 10 (life-changing). Return a JSON object.",
        "Score how important this event is to the agent's life.\n"
        "1 = routine/mundane, 5 = notable, 10 = life-changing.\n"
        "Output: JSON with score (integer 1-10).\n\n"
    ),
    "conversation": (
        "You are a V-Valley conversation generator. Each agent has a persona with innate traits, backstory, "
        "and occupation — use these to make dialogue feel authentic and in-character. Create a realistic, "
        "memory-grounded conversation between two agents. Return a JSON object with utterances and summary.",
        "Generate a natural conversation between these two agents based on their personas, memories and activities.\n"
        "Output: JSON with utterances (array of {speaker, text}) and summary (string).\n\n"
    ),
    "focal_points": (
        "You are a V-Valley reflection focal point generator. Given recent memory statements, "
        "generate 1-3 high-level questions worth reflecting on. Return a JSON object.",
        "What are the most salient high-level questions the agent should reflect on?\n"
        "Output: JSON with questions (array of strings, 1-3 questions).\n\n"
    ),
    "reflection_insights": (
        "You are a V-Valley reflection insight synthesizer. Given a focal question and retrieved "
        "memory statements, generate insights with evidence. Return a JSON object.",
        "Synthesize insights from the retrieved memories about the focal point.\n"
        "Output: JSON with insights (array of {insight, evidence_indices}).\n\n"
    ),
    "schedule_decomp": (
        "You are a V-Valley schedule decomposer. Break a schedule block into smaller subtasks "
        "that sum to the total duration. Return a JSON object.",
        "Decompose this activity into detailed minute-level subtasks.\n"
        "The subtask durations must sum to approximately the total duration.\n"
        "Output: JSON with subtasks (array of {description, duration_mins}).\n\n"
    ),
    "conversation_summary": (
        "You are a V-Valley conversation summarizer. Generate a takeaway memo and a planning thought "
        "from a completed conversation. Return a JSON object.",
        "Summarize what the agent should take away from this conversation.\n"
        "Output: JSON with memo (string, key takeaway) and planning_thought (string, what to do next).\n\n"
    ),
    "conversation_turn": (
        "You are a V-Valley conversation agent. Each character has a persona with specific personality traits, "
        "occupation, and background. Generate the next utterance staying true to their personality "
        "and using their memories for context. Set end_conversation to true if the conversation should naturally end. "
        "Return a JSON object.",
        "Generate the next line of dialogue for this character, staying in character based on their persona "
        "and using their memories for context. Set end_conversation=true if the exchange has reached a natural end.\n"
        "Output: JSON with utterance (string) and end_conversation (boolean).\n\n"
    ),
    "daily_schedule_gen": (
        "You are a V-Valley daily schedule planner. Generate a full day schedule for an agent "
        "based on their persona (innate traits, occupation, lifestyle) and recent experiences. Also update their 'currently' "
        "status. Return a JSON object.",
        "Generate a realistic daily schedule reflecting the agent's persona, occupation, and lifestyle "
        "(activities with durations summing to 1440 minutes). "
        "Also provide a short 'currently' string describing the agent's identity and focus.\n"
        "Output: JSON with schedule (array of {description, duration_mins}) and currently (string).\n\n"
    ),
    "persona_gen": (
        "You are a character creator for V-Valley, a small-town life simulation. Generate a unique, "
        "believable character persona for a resident of a cozy pixel-art town. The character should feel "
        "like a real person with specific habits, personality quirks, and a concrete daily routine. "
        "Return a JSON object.",
        "Create a complete character persona. If a name is provided in the context, use it. "
        "Otherwise invent a fitting full name.\n"
        "Output: JSON with these exact fields:\n"
        "- first_name (string)\n"
        "- last_name (string)\n"
        "- age (integer, 18-75)\n"
        "- innate (string, 3 comma-separated personality traits like 'friendly, curious, determined')\n"
        "- learned (string, 1-2 sentence third-person backstory/biography)\n"
        "- lifestyle (string, sleep/wake habits like 'goes to bed around 11pm, wakes at 7am')\n"
        "- daily_plan_req (string, numbered daily plan)\n"
        "- daily_req (array of 5-7 short activity strings with times)\n\n"
    ),
    "pronunciatio": (
        "You are a V-Valley action emoji tagger. Given an agent's current action description, "
        "return a single emoji that best represents the action, and a subject-predicate-object "
        "event triple. Return a JSON object.",

        "What emoji best represents this action? Also extract an event triple.\n"
        "Output: JSON with emoji (string, single emoji character), "
        "event_triple (object with subject, predicate, object strings).\n\n"
    ),
    "wake_up_hour": (
        "You are a V-Valley daily routine analyzer. Given an agent's lifestyle description "
        "and personality, determine what hour they typically wake up. Return a JSON object.",

        "Based on this agent's lifestyle, what hour (0-23) do they typically wake up?\n"
        "Output: JSON with wake_up_hour (integer 0-23).\n\n"
    ),
    "action_address": (
        "You are a V-Valley spatial planner. Given an agent's current action and the available "
        "locations in town, choose the best sector, arena (room), and game object where this "
        "action should take place. Consider the agent's identity and living area. Return JSON.",

        "Where should this agent go to perform their current action?\n"
        "Choose from the available sectors, arenas, and objects.\n"
        "Output: JSON with action_sector (string), action_arena (string or null), "
        "action_game_object (string or null).\n\n"
    ),
    "object_state": (
        "You are a V-Valley object state tracker. When an agent uses a game object, "
        "describe the current state of that object. Return JSON.",

        "Describe the state of this object while the agent is using it.\n"
        "Output: JSON with object_description (string), "
        "object_event_triple (object with subject, predicate, object).\n\n"
    ),
    "event_reaction": (
        "You are a V-Valley event reaction analyzer. Given a perceived event and the agent's "
        "current activity, decide if the agent should react by changing their plan. "
        "Agents owned by users should stay true to their personality. Return JSON.",

        "Should this agent react to the observed event by changing their current plan?\n"
        "Output: JSON with react (boolean), reason (string), "
        "new_action (string, only if react=true).\n\n"
    ),
    "relationship_summary": (
        "You are a V-Valley relationship narrator. Given two agents' identities and their "
        "interaction history, write a brief narrative summary of their relationship "
        "to ground a conversation. Return JSON.",

        "Summarize the relationship between these two agents before they start talking.\n"
        "Output: JSON with summary (string, 1-2 sentences), "
        "conversation_tone (string: warm/polite/cautious/playful).\n\n"
    ),
    "first_daily_plan": (
        "You are a V-Valley first-day planner. This agent just arrived in the town. "
        "Generate an exploratory daily schedule that helps them settle in, explore, "
        "and meet neighbors. Consider their personality and background. Return JSON.",

        "Create a first-day schedule for this newly arrived agent.\n"
        "Include exploration, settling in, and socializing.\n"
        "Output: JSON with schedule (array of {description, duration_mins}), "
        "currently (string describing their initial focus).\n\n"
    ),
    "schedule_recomposition": (
        "You are a V-Valley schedule recomposer. An agent's current plan was interrupted "
        "by a reaction or conversation. Rewrite the remainder of their daily schedule "
        "to accommodate the interrupting activity while preserving as much of the "
        "original plan as possible. Total schedule should be ~1440 minutes. Return JSON.",

        "Recompose this agent's daily schedule after an interruption.\n"
        "Insert the new activity at the current time, then adjust remaining blocks.\n"
        "Output: JSON with schedule (array of {description, duration_mins}).\n\n"
    ),
}

_DEFAULT_PROMPT = (
    "You are the V-Valley cognition engine. "
    "Return a strict JSON object only, without markdown or commentary. "
    "Never wrap JSON in code fences.",
    "Return a JSON object for this task.\n\n",
)


def _build_prompt(*, task_name: str, bounded_context_text: str) -> list[dict[str, str]]:
    system_text, user_prefix = _TASK_PROMPTS.get(task_name, _DEFAULT_PROMPT)
    system = f"{system_text} Return strict JSON only, no markdown or code fences."
    user = f"{user_prefix}Context JSON:\n{bounded_context_text}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_content_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        chunks: list[str] = []
        for item in message:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    chunks.append(str(text))
        return "".join(chunks)
    return str(message or "")


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = str(text or "").strip()
    if not candidate:
        raise ProviderExecutionError("Empty model response", error_code="empty_response")

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    in_string = False
    escape = False
    depth = 0
    start = None
    for idx, char in enumerate(candidate):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if char == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                blob = candidate[start : idx + 1]
                try:
                    parsed = json.loads(blob)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    return parsed
                break

    raise ProviderExecutionError(
        "Response does not contain a valid JSON object",
        error_code="invalid_json_output",
    )


def _post_chat_completions(
    *,
    config: TierProviderConfig,
    payload: dict[str, Any],
    timeout_ms: int,
) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = request.Request(
        f"{config.base_url}/chat/completions",
        method="POST",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    if config.api_key:
        req.add_header("Authorization", f"Bearer {config.api_key}")

    timeout_s = max(0.2, float(timeout_ms) / 1000.0)
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        raise ProviderExecutionError(
            f"Provider HTTP error {exc.code}: {detail[:240]}",
            error_code=f"http_{exc.code}",
            model_name=config.model_name(),
        ) from exc
    except Exception as exc:
        raise ProviderExecutionError(
            f"Provider network error: {exc}",
            error_code="network_error",
            model_name=config.model_name(),
        ) from exc

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise ProviderExecutionError(
            "Provider returned non-JSON response",
            error_code="invalid_provider_response",
            model_name=config.model_name(),
        ) from exc
    if not isinstance(parsed, dict):
        raise ProviderExecutionError(
            "Provider returned invalid JSON payload",
            error_code="invalid_provider_response",
            model_name=config.model_name(),
        )
    return parsed


def _extract_content_text_from_response(parsed: dict[str, Any], *, model_name: str) -> str:
    choices = parsed.get("choices") or []
    if not choices:
        raise ProviderExecutionError(
            "Provider response missing choices",
            error_code="missing_choices",
            model_name=model_name,
        )
    first = choices[0] or {}
    message = (first.get("message") or {}).get("content")
    return _parse_content_text(message)


def _repair_json_output_with_model(
    *,
    config: TierProviderConfig,
    task_name: str,
    malformed_output: str,
    max_output_tokens: int,
    timeout_ms: int,
) -> dict[str, Any]:
    schema = _schema_for_task(task_name)
    schema_hint = ""
    if schema:
        schema_hint = "\nTarget JSON schema:\n" + json.dumps(schema, separators=(",", ":"))[:2000]
    repair_prompt = [
        {
            "role": "system",
            "content": "Rewrite the input as one valid JSON object only. No markdown.",
        },
        {
            "role": "user",
            "content": (
                f"Task: {task_name}\n"
                "Repair this into valid JSON object output only:\n"
                f"{str(malformed_output or '')[:4000]}"
                f"{schema_hint}"
            ),
        },
    ]
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": repair_prompt,
        "temperature": 0.0,
        "max_tokens": max(64, min(512, int(max_output_tokens))),
    }
    response_format = _response_format_for_task(task_name)
    if response_format:
        payload["response_format"] = response_format
    parsed = _post_chat_completions(config=config, payload=payload, timeout_ms=timeout_ms)
    repaired_text = _extract_content_text_from_response(parsed, model_name=config.model_name())
    return _extract_json_object(repaired_text)


def _post_openai_compatible(
    *,
    config: TierProviderConfig,
    task_name: str,
    bounded_context_text: str,
    temperature: float,
    max_output_tokens: int,
    timeout_ms: int,
) -> ProviderExecutionResult:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": _build_prompt(task_name=task_name, bounded_context_text=bounded_context_text),
        "temperature": float(temperature),
        "max_tokens": int(max_output_tokens),
    }
    response_format = _response_format_for_task(task_name)
    if response_format:
        payload["response_format"] = response_format

    try:
        parsed = _post_chat_completions(config=config, payload=payload, timeout_ms=timeout_ms)
    except ProviderExecutionError as exc:
        # Some OpenAI-compatible providers do not support `response_format`.
        if response_format and exc.error_code == "http_400" and any(
            hint in str(exc).lower() for hint in ("response_format", "json_schema", "additionalproperties", "strict")
        ):
            payload.pop("response_format", None)
            parsed = _post_chat_completions(config=config, payload=payload, timeout_ms=timeout_ms)
        else:
            raise

    content_text = _extract_content_text_from_response(parsed, model_name=config.model_name())
    try:
        output = _extract_json_object(content_text)
        _validate_output_schema(task_name, output)
    except ProviderExecutionError as exc:
        if exc.error_code not in {"invalid_json_output", "invalid_schema_output"}:
            raise
        try:
            output = _repair_json_output_with_model(
                config=config,
                task_name=task_name,
                malformed_output=content_text,
                max_output_tokens=max_output_tokens,
                timeout_ms=timeout_ms,
            )
            _validate_output_schema(task_name, output)
        except Exception:
            raise exc

    usage = parsed.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    try:
        prompt = int(prompt_tokens) if prompt_tokens is not None else estimate_token_count(bounded_context_text)
    except Exception:
        prompt = estimate_token_count(bounded_context_text)
    try:
        completion = (
            int(completion_tokens)
            if completion_tokens is not None
            else estimate_token_count(json.dumps(output, separators=(",", ":")))
        )
    except Exception:
        completion = estimate_token_count(json.dumps(output, separators=(",", ":")))

    return ProviderExecutionResult(
        output=output,
        model_name=str(parsed.get("model") or config.model_name()),
        prompt_tokens=prompt,
        completion_tokens=completion,
    )


def execute_tier_model(
    *,
    tier: str,
    task_name: str,
    bounded_context_text: str,
    temperature: float,
    max_output_tokens: int,
    timeout_ms: int,
) -> ProviderExecutionResult:
    """Execute a policy task against the configured provider for one tier."""
    config = _tier_provider_config(tier)
    return _post_openai_compatible(
        config=config,
        task_name=task_name,
        bounded_context_text=bounded_context_text,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout_ms=timeout_ms,
    )


def generate_persona(*, name: str | None = None, tier: str = "fast") -> dict[str, Any]:
    """Generate a full GA-format character persona using the configured LLM.

    Returns a dict with: first_name, last_name, age, innate, learned,
    lifestyle, daily_plan_req, daily_req.
    """
    context = {}
    if name and name.strip():
        context["name"] = name.strip()
    else:
        context["instruction"] = "Invent a unique full name for this character."

    result = execute_tier_model(
        tier=tier,
        task_name="persona_gen",
        bounded_context_text=json.dumps(context, separators=(",", ":")),
        temperature=0.9,
        max_output_tokens=800,
        timeout_ms=30_000,
    )
    persona = result.output

    # Ensure required fields exist with fallbacks
    if "first_name" not in persona:
        persona["first_name"] = name.split()[0] if name else "Unknown"
    if "last_name" not in persona:
        parts = (name or "").split()
        persona["last_name"] = parts[-1] if len(parts) > 1 else "Resident"
    if "age" not in persona:
        persona["age"] = 30
    if "innate" not in persona:
        persona["innate"] = "curious, friendly, adaptable"
    if "learned" not in persona:
        full_name = f"{persona['first_name']} {persona['last_name']}"
        persona["learned"] = f"{full_name} is a resident of the town."
    if "lifestyle" not in persona:
        persona["lifestyle"] = "goes to bed around 11pm, wakes up around 7am"
    if "daily_plan_req" not in persona:
        persona["daily_plan_req"] = ""
    if "daily_req" not in persona:
        persona["daily_req"] = []

    return persona
