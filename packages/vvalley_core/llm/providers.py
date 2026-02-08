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
        "Return a strict JSON object only, without markdown or commentary."
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


def _post_openai_compatible(
    *,
    config: TierProviderConfig,
    task_name: str,
    bounded_context_text: str,
    temperature: float,
    max_output_tokens: int,
    timeout_ms: int,
) -> ProviderExecutionResult:
    payload = {
        "model": config.model,
        "messages": _build_prompt(task_name=task_name, bounded_context_text=bounded_context_text),
        "temperature": float(temperature),
        "max_tokens": int(max_output_tokens),
    }
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

    choices = parsed.get("choices") or []
    if not choices:
        raise ProviderExecutionError(
            "Provider response missing choices",
            error_code="missing_choices",
            model_name=config.model_name(),
        )
    first = choices[0] or {}
    message = (first.get("message") or {}).get("content")
    content_text = _parse_content_text(message)
    output = _extract_json_object(content_text)

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

