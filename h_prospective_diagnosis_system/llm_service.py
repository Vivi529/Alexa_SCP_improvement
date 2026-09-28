from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import httpx
from openai import OpenAI

def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or '').strip()
    if not text:
        raise ValueError('Empty LLM response')
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, flags=re.S)
    if not match:
        raise ValueError('No JSON object found in LLM response')
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError('LLM response JSON must be an object')
    return obj


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return bool(default)

    return raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _build_client(
    *,
    api_key: str,
    base_url: str,
) -> OpenAI:

    trust_env = _env_bool(
        "LLM_TRUST_ENV_PROXY",
        default=False,
    )

    max_retries = int(
        os.getenv(
            "LLM_HTTP_MAX_RETRIES",
            "0",
        )
    )

    http_client = httpx.Client(
        trust_env=trust_env,
        timeout=httpx.Timeout(
            connect=30.0,
            read=120.0,
            write=30.0,
            pool=30.0,
        ),
    )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
        http_client=http_client,
    )


def call_json(
    system_prompt: str,
    user_payload: Dict[str, Any],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    use_json_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    OpenAI-compatible JSON call.

    Exactly one API request is issued per call_json invocation.

    Environment variables:
      LLM_API_KEY        required
      LLM_BASE_URL       optional
      LLM_MODEL          optional if model_name is passed
      LLM_USE_JSON_MODE  optional: 1/true/yes/on to enable response_format=json_object
    """

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set LLM_API_KEY before calling the model."
        )

    base_url = os.getenv("LLM_BASE_URL") or None

    model = (
        model_name
        or os.getenv("LLM_MODEL")
    )

    if not model:
        raise RuntimeError(
            "Set LLM_MODEL or pass model_name."
        )

    # ---------------------------------------------------------
    # Resolve JSON mode once, before sending the request.
    #
    # IMPORTANT:
    # We do NOT retry automatically without response_format.
    # One call_json invocation = one model API request.
    # ---------------------------------------------------------

    if use_json_mode is None:
        raw_flag = str(
            os.getenv(
                "LLM_USE_JSON_MODE",
                "true",
            )
        ).strip().lower()

        use_json_mode = (
            raw_flag
            in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
        )

    # Disable SDK-level automatic HTTP retries for controlled experiments.
    # This makes one call_json invocation correspond to one SDK request attempt.  
    client = _build_client(
        api_key=api_key,
        base_url=base_url,
    )

    # Compact JSON reduces prompt-token overhead.
    payload_text = json.dumps(
        user_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    kwargs = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": payload_text,
            },
        ],
    }

    # ---------------------------------------------------------
    # Exactly ONE API request
    # ---------------------------------------------------------

    if use_json_mode:
        resp = client.chat.completions.create(
            **kwargs,
            response_format={
                "type": "json_object",
            },
        )
    else:
        resp = client.chat.completions.create(
            **kwargs,
        )

    # ---------------------------------------------------------
    # Extract raw response
    # ---------------------------------------------------------

    text = (
        resp.choices[0].message.content
        or ""
    )

    usage_obj = getattr(
        resp,
        "usage",
        None,
    )

    def _usage_value(name: str):
        if usage_obj is None:
            return None

        value = getattr(
            usage_obj,
            name,
            None,
        )

        try:
            return (
                None
                if value is None
                else int(value)
            )
        except Exception:
            return None

    # ---------------------------------------------------------
    # Prompt-cache token accounting
    # ---------------------------------------------------------

    cached_tokens = None

    prompt_details = (
        getattr(
            usage_obj,
            "prompt_tokens_details",
            None,
        )
        if usage_obj is not None
        else None
    )

    if prompt_details is not None:
        value = getattr(
            prompt_details,
            "cached_tokens",
            None,
        )

        try:
            cached_tokens = (
                None
                if value is None
                else int(value)
            )
        except Exception:
            cached_tokens = None

    # ---------------------------------------------------------
    # Preserve raw response even if JSON parsing fails.
    #
    # Parsing failure does NOT trigger another API request.
    # ---------------------------------------------------------

    parse_error = None

    try:
        parsed = _extract_json(text)

    except Exception as exc:
        parsed = None
        parse_error = repr(exc)

    return {
        "parsed": parsed,
        "raw_output": text,
        "parse_error": parse_error,

        "usage": {
            "prompt_tokens":
                _usage_value(
                    "prompt_tokens"
                ),

            "cached_prompt_tokens":
                cached_tokens,

            "completion_tokens":
                _usage_value(
                    "completion_tokens"
                ),

            "total_tokens":
                _usage_value(
                    "total_tokens"
                ),

            "system_prompt_chars":
                len(
                    system_prompt
                    or ""
                ),

            "user_payload_chars":
                len(
                    payload_text
                ),
        },

        # Useful for later audit
        "request_meta": {
            "model": model,
            "json_mode": bool(use_json_mode),
            "sdk_max_retries": 0,
        },
    }