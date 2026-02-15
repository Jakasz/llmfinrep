"""Parse and validate JSON response from LLM extraction step."""

import json
import logging
import re

logger = logging.getLogger(__name__)

# Required keys in the extracted data
REQUIRED_KEYS = {"balance_start", "balance_end", "income_current"}

# Known balance row codes
BALANCE_CODES = {
    "1002", "1011", "1012", "1095", "1100",
    "1125", "1135", "1155", "1160", "1165", "1195",
    "1495", "1595", "1600", "1610",
    "1615", "1620", "1625", "1690", "1695", "1900",
}

# Known income statement codes
INCOME_CODES = {"2000", "2050", "2290", "2350"}


class ExtractionError(Exception):
    """Raised when LLM response cannot be parsed as valid financial data."""


def parse_llm_json(raw_response: str) -> dict:
    """Parse LLM response text into a validated dict of financial data.

    Handles common LLM quirks:
    - Leading/trailing text around JSON
    - Markdown code fences (```json ... ```)
    - String values that should be numeric ("732,8" -> 732.8)

    Args:
        raw_response: Raw text from LLM.

    Returns:
        Dict with keys: company_name, period, balance_start, balance_end, income_current.

    Raises:
        ExtractionError: If response cannot be parsed or is missing required data.
    """
    if not raw_response or not raw_response.strip():
        raise ExtractionError("LLM повернув порожню відповідь")

    # Step 1: Extract JSON from possible markdown fences or surrounding text
    json_str = _extract_json_block(raw_response)

    # Step 2: Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("JSON parse error: %s", e)
        logger.debug("Raw response (first 500 chars): %s", raw_response[:500])
        raise ExtractionError(f"Не вдалося розпарсити JSON: {e}") from e

    if not isinstance(data, dict):
        raise ExtractionError(f"Очікувався JSON-об'єкт, отримано: {type(data).__name__}")

    # Step 3: Validate required keys
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ExtractionError(f"Відсутні обов'язкові ключі: {', '.join(sorted(missing))}")

    # Step 4: Normalize numeric values
    for key in ("balance_start", "balance_end", "income_current"):
        section = data.get(key, {})
        if not isinstance(section, dict):
            raise ExtractionError(f"Ключ '{key}' має бути об'єктом, отримано: {type(section).__name__}")
        data[key] = _normalize_section(section)

    # Step 5: Ensure string fields
    data.setdefault("company_name", "Невідома компанія")
    data.setdefault("period", "—")

    # Log summary
    bs = data["balance_start"]
    be = data["balance_end"]
    ic = data["income_current"]
    logger.info(
        "Extracted data: company=%s, period=%s, "
        "balance_start=%d codes, balance_end=%d codes, income=%d codes",
        data["company_name"], data["period"], len(bs), len(be), len(ic),
    )

    return data


def _extract_json_block(text: str) -> str:
    """Extract JSON object from text that may contain markdown fences or extra text."""
    text = text.strip()

    # Try markdown code fence: ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Try to find JSON object by matching braces
    first_brace = text.find("{")
    if first_brace == -1:
        raise ExtractionError("JSON-об'єкт не знайдено у відповіді LLM")

    # Find matching closing brace
    depth = 0
    for i in range(first_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[first_brace:i + 1]

    # If no matching brace found, try the whole thing from first brace
    return text[first_brace:]


def _normalize_section(section: dict) -> dict:
    """Normalize keys and values in a section dict.

    - Keys: ensure they are strings of digits
    - Values: convert string numbers to float, handle Ukrainian comma decimal
    """
    result = {}
    for key, val in section.items():
        # Normalize key — strip "р." prefix if present
        k = str(key).strip()
        k = re.sub(r"^р\.?", "", k).strip()
        if not k.isdigit():
            logger.warning("Skipping non-numeric key: %s", key)
            continue

        # Normalize value
        numeric = _to_float(val)
        if numeric is not None:
            result[k] = numeric
        else:
            logger.warning("Could not convert value for code %s: %r", k, val)

    return result


def _to_float(val) -> float | None:
    """Convert a value to float, handling various formats."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if not s or s == "-" or s.lower() == "null" or s.lower() == "none":
            return None
        # Handle parentheses as negative: (123.4) -> -123.4
        paren_match = re.match(r"^\((.+)\)$", s)
        if paren_match:
            inner = _to_float(paren_match.group(1))
            return -inner if inner is not None else None
        # Replace comma decimal separator
        s = s.replace(",", ".").replace(" ", "").replace("\u00a0", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None
