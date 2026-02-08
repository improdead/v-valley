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
        "additionalProperties": True,
    },
    "short_action": {
        "type": "object",
        "properties": {
            "dx": {"type": "integer", "minimum": -1, "maximum": 1},
            "dy": {"type": "integer", "minimum": -1, "maximum": 1},
            "reason": {"type": "string", "maxLength": 200},
        },
        "required": ["dx", "dy", "reason"],
        "additionalProperties": True,
    },
    "reaction_policy": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["talk", "wait", "ignore"]},
            "reason": {"type": "string", "maxLength": 240},
            "wait_steps": {"type": "integer", "minimum": 0, "maximum": 8},
        },
        "required": ["decision", "reason"],
        "additionalProperties": True,
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
            "additionalProperties": True,
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


def _build_prompt(*, task_name: str, bounded_context_text: str) -> list[dict[str, str]]:
    system = (
        "You are the V-Valley cognition engine. "
        "Return a strict JSON object only, without markdown or commentary. "
        "Never wrap JSON in code fences."
    )
    user = (
        f"Task: {task_name}\n"
        "Output contract:\n"
        "- Always return a JSON object.\n"
        "- For movement-related tasks, include integer dx and dy in range -1..1 and a short reason string.\n\n"
        "Context JSON:\n"
        f"{bounded_context_text}"
    )
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
        if response_format and exc.error_code == "http_400" and "response_format" in str(exc).lower():
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
