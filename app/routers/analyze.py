"""POST /api/v1/analyze endpoint — 3-step hybrid pipeline.

Step 1: LLM extracts row values from documents → JSON
Step 2: Python calculates all financial coefficients
Step 3: LLM generates HTML report from pre-calculated data
"""

import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.auth import verify_api_key
from app.config import AppConfig, get_config
from app.services.extractor import ALLOWED_EXTENSIONS, extract_text, is_allowed_extension
from app.services.financial_calculator import CalculationResult, calculate_all
from app.services.json_extractor import ExtractionError, parse_llm_json
from app.services.llm_client import query_ollama_json, query_ollama_report
from app.utils.text_utils import combine_extracted_texts, estimate_tokens, truncate_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analysis"])

MAX_FILES = 10


def _load_prompt(path_str: str) -> str:
    """Load a prompt template file."""
    prompt_path = Path(path_str)
    if not prompt_path.is_absolute():
        prompt_path = Path(__file__).parent.parent.parent / prompt_path
    if not prompt_path.exists():
        raise RuntimeError(f"Prompt template not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def _format_calculations_for_llm(calc: CalculationResult) -> str:
    """Format CalculationResult as text for LLM report generation."""
    lines = []
    lines.append(f"Компанія: {calc.company_name}")
    lines.append(f"Період: {calc.period}")
    lines.append("")

    for i, section in enumerate(calc.sections, 1):
        lines.append(f"{i}. {section.title}")
        if not section.rows:
            lines.append("   (немає даних для розрахунку)")
        for row in section.rows:
            result_str = f"{row.result}" if isinstance(row.result, str) else f"{row.result:.4f}"
            lines.append(
                f"   - {row.name}: {row.formula} = {result_str} "
                f"[{row.rating}:{row.rating_label}] (норма: {row.norm})"
            )
        lines.append("")

    if calc.limitations:
        lines.append("ОБМЕЖЕННЯ:")
        for lim in calc.limitations:
            lines.append(f"   - {lim.indicator}: {lim.reason} ({lim.missing_rows})")
    else:
        lines.append("ОБМЕЖЕННЯ: немає — усі показники розраховано.")

    return "\n".join(lines)


@router.post("/analyze")
async def analyze_documents(
    files: list[UploadFile] = File(...),
    user_instructions: str = Form(""),
    _api_key: str = Depends(verify_api_key),
    config: AppConfig = Depends(get_config),
):
    """Accept uploaded documents, extract text, and return LLM financial analysis.

    Pipeline:
      1. Extract text from files
      2. LLM extracts row values → JSON
      3. Python calculates coefficients
      4. LLM generates HTML report
    """
    start_time = time.time()

    # --- File validation ---
    if len(files) < 1 or len(files) > MAX_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Please upload between 1 and {MAX_FILES} files. Received: {len(files)}",
        )

    for f in files:
        if not is_allowed_extension(f.filename or ""):
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unsupported file type: '{f.filename}'. Allowed: {allowed}",
            )

    # --- Read files and check size ---
    max_size = config.max_upload_size_mb * 1024 * 1024
    file_data: list[tuple[str, bytes]] = []
    total_size = 0

    for f in files:
        content = await f.read()
        total_size += len(content)
        if total_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Total upload size exceeds {config.max_upload_size_mb} MB limit",
            )
        file_data.append((f.filename or "unknown", content))

    # --- Extract text from each file ---
    file_texts: dict[str, str] = {}
    failed_files: list[dict] = []

    for filename, content in file_data:
        try:
            text = extract_text(filename, content)
            file_texts[filename] = text
        except Exception as e:
            logger.error("Failed to extract text from '%s': %s", filename, e, exc_info=True)
            failed_files.append({"filename": filename, "error": str(e)})

    if not file_texts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Failed to extract text from all uploaded files",
                "failed_files": failed_files,
            },
        )

    # --- Combine texts ---
    combined_text = combine_extracted_texts(file_texts)
    tokens_estimated = estimate_tokens(combined_text)
    logger.info("Combined text: %d characters, ~%d tokens", len(combined_text), tokens_estimated)

    was_truncated = False
    if tokens_estimated > config.max_total_tokens_estimate:
        combined_text, was_truncated = truncate_text(combined_text, config.max_total_tokens_estimate)
        tokens_estimated = estimate_tokens(combined_text)

    # =====================================================
    # STEP 1: LLM extracts row values → JSON
    # =====================================================
    step1_start = time.time()
    logger.info("=== STEP 1: Data extraction (LLM → JSON) ===")

    try:
        extraction_prompt = _load_prompt(config.extraction_prompt_file)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Extraction prompt error: {e}")

    # Build the extraction request
    extraction_user_content = extraction_prompt.replace("{documents}", combined_text)

    # System prompt is the part before "--- ДОКУМЕНТИ ---"
    extraction_system = ""
    if "--- ДОКУМЕНТИ ---" in extraction_prompt:
        extraction_system = extraction_prompt.split("--- ДОКУМЕНТИ ---")[0].strip()
        extraction_user_content = f"--- ДОКУМЕНТИ ---\n{combined_text}"

    try:
        raw_json_response = await query_ollama_json(extraction_system, extraction_user_content, config)
    except Exception as e:
        logger.error("Step 1 failed (LLM extraction): %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Data extraction failed: {e}")

    step1_time = round(time.time() - step1_start, 2)
    logger.info("Step 1 completed in %.2fs, response: %d chars", step1_time, len(raw_json_response))

    # Parse JSON response
    try:
        extracted_data = parse_llm_json(raw_json_response)
    except ExtractionError as e:
        logger.error("Step 1 JSON parsing failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Failed to parse extracted data: {e}",
                "raw_response_preview": raw_json_response[:1000] if raw_json_response else "",
            },
        )

    # =====================================================
    # STEP 2: Python calculates all coefficients
    # =====================================================
    step2_start = time.time()
    logger.info("=== STEP 2: Python financial calculations ===")

    calc_result = calculate_all(extracted_data)
    calculations_text = _format_calculations_for_llm(calc_result)

    step2_time = round(time.time() - step2_start, 4)
    logger.info("Step 2 completed in %.4fs", step2_time)

    # =====================================================
    # STEP 3: LLM generates HTML report
    # =====================================================
    step3_start = time.time()
    logger.info("=== STEP 3: Report generation (LLM → HTML) ===")

    try:
        report_prompt = _load_prompt(config.report_prompt_file)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Report prompt error: {e}")

    # Build report request
    report_system = ""
    if "--- РОЗРАХУНКИ ---" in report_prompt:
        report_system = report_prompt.split("--- РОЗРАХУНКИ ---")[0].strip()
        report_user_content = f"--- РОЗРАХУНКИ ---\n{calculations_text}"
    else:
        report_user_content = report_prompt.replace("{calculations}", calculations_text)

    if user_instructions.strip():
        report_user_content += f"\n\n--- ДОДАТКОВІ ІНСТРУКЦІЇ ---\n{user_instructions.strip()}"

    try:
        report_html = await query_ollama_report(report_system, report_user_content, config)
    except Exception as e:
        logger.error("Step 3 failed (report generation): %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    step3_time = round(time.time() - step3_start, 2)
    logger.info("Step 3 completed in %.2fs, report: %d chars", step3_time, len(report_html))

    # Post-process: replace \n with <br> for proper HTML rendering
    report_html = report_html.replace("\n", "<br>\n")

    # =====================================================
    # Response
    # =====================================================
    processing_time = round(time.time() - start_time, 2)
    logger.info("Full pipeline completed in %.2fs (step1=%.2fs, step2=%.4fs, step3=%.2fs)",
                processing_time, step1_time, step2_time, step3_time)

    response = {
        "status": "success",
        "report": report_html,
        "extracted_data": extracted_data,
        "files_processed": list(file_texts.keys()),
        "tokens_estimated": tokens_estimated,
        "processing_time_seconds": processing_time,
        "pipeline_steps": {
            "extraction_seconds": step1_time,
            "calculation_seconds": step2_time,
            "report_seconds": step3_time,
        },
    }

    if failed_files:
        response["failed_files"] = failed_files
    if was_truncated:
        response["warning"] = "Document text was truncated due to token limit"

    return response
