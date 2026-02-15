"""Ollama LLM API client (async, with timeout and error handling).

Uses the /api/chat endpoint with configurable model, context window, and temperature.
"""

import json as json_module
import logging

import httpx

from app.config import AppConfig

logger = logging.getLogger(__name__)


async def _send_ollama_request(
    system_prompt: str,
    user_content: str,
    config: AppConfig,
    *,
    num_predict_override: int | None = None,
    temperature_override: float | None = None,
    json_format: bool = False,
    step_name: str = "",
) -> str:
    """Internal: send a chat request to Ollama and return the content.

    Args:
        system_prompt: System-level instructions.
        user_content: User message.
        config: App config.
        num_predict_override: Override num_predict from config.
        temperature_override: Override temperature from config.
        json_format: If True, request JSON output format from Ollama.
        step_name: Label for logging (e.g. "extraction", "report").

    Returns:
        LLM-generated response text.
    """
    url = f"{config.ollama_base_url}/api/chat"
    timeout = httpx.Timeout(timeout=float(config.ollama_timeout), connect=30.0)

    options = {
        "num_ctx": config.ollama_num_ctx,
        "temperature": temperature_override if temperature_override is not None else config.ollama_temperature,
        "num_predict": num_predict_override if num_predict_override is not None else config.ollama_num_predict,
        "repeat_penalty": config.ollama_repeat_penalty,
        "repeat_last_n": config.ollama_repeat_last_n,
    }

    payload = {
        "model": config.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": options,
    }

    if json_format:
        payload["format"] = "json"

    label = f"[{step_name}] " if step_name else ""
    logger.info(
        "%sSending request to Ollama (model=%s, num_ctx=%d, num_predict=%d, temp=%.1f, json=%s, timeout=%ds)",
        label, config.ollama_model, options["num_ctx"], options["num_predict"],
        options["temperature"], json_format, config.ollama_timeout,
    )
    logger.debug("%sPayload user content length: %d chars", label, len(user_content))

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()

    data = response.json()

    # Diagnostics
    done_reason = data.get("done_reason", "unknown")
    eval_count = data.get("eval_count", 0)
    prompt_eval_count = data.get("prompt_eval_count", 0)
    logger.info(
        "%sOllama response meta: done_reason=%s, prompt_tokens=%d, eval_tokens=%d",
        label, done_reason, prompt_eval_count, eval_count,
    )

    message = data.get("message", {})
    content = message.get("content", "")
    thinking = message.get("thinking", "")

    if thinking:
        logger.info("%sOllama thinking: %d characters", label, len(thinking))

    if not content and thinking:
        logger.warning(
            "%sContent is EMPTY but thinking has %d chars. "
            "done_reason=%s — збільшіть num_predict в config.ini",
            label, len(thinking), done_reason,
        )

    if not content and not thinking:
        logger.warning("%sOllama returned EMPTY content and no thinking.", label)
        logger.warning("%sResponse (truncated): %s", label, json_module.dumps(data, ensure_ascii=False)[:500])

    logger.info("%sOllama response received: %d characters", label, len(content))
    return content


async def query_ollama(
    system_prompt: str,
    user_content: str,
    config: AppConfig,
) -> str:
    """Send a chat request to the Ollama API and return the response (legacy API)."""
    return await _send_ollama_request(system_prompt, user_content, config, step_name="legacy")


async def query_ollama_json(
    system_prompt: str,
    user_content: str,
    config: AppConfig,
) -> str:
    """Send a chat request expecting JSON response.

    Uses lower temperature (0.1) and JSON format mode.
    num_predict is set to 4096 — sufficient for structured data extraction.
    """
    return await _send_ollama_request(
        system_prompt,
        user_content,
        config,
        num_predict_override=4096,
        temperature_override=0.1,
        json_format=True,
        step_name="extraction",
    )


async def query_ollama_report(
    system_prompt: str,
    user_content: str,
    config: AppConfig,
) -> str:
    """Send a chat request for HTML report generation.

    Uses the configured num_predict and temperature.
    """
    return await _send_ollama_request(
        system_prompt,
        user_content,
        config,
        step_name="report",
    )


async def check_ollama_health(config: AppConfig) -> dict:
    """Check Ollama API connectivity and model availability.

    Returns:
        Dict with status information.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            # Перевірка доступності Ollama
            resp = await client.get(f"{config.ollama_base_url}/api/tags")
            resp.raise_for_status()
            tags_data = resp.json()

            # Перевірка наявності потрібної моделі
            available_models = [m["name"] for m in tags_data.get("models", [])]
            model_available = any(
                config.ollama_model in model_name
                for model_name in available_models
            )

            return {
                "ollama_reachable": True,
                "model_available": model_available,
                "configured_model": config.ollama_model,
                "available_models": available_models,
            }
    except httpx.ConnectError:
        return {
            "ollama_reachable": False,
            "model_available": False,
            "configured_model": config.ollama_model,
            "error": f"Cannot connect to Ollama at {config.ollama_base_url}",
        }
    except Exception as e:
        return {
            "ollama_reachable": False,
            "model_available": False,
            "configured_model": config.ollama_model,
            "error": str(e),
        }
