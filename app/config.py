"""Application configuration reader.

Reads settings from config.ini using configparser.
All settings have sensible defaults if keys are missing.
"""

import configparser
import logging
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).parent.parent / "config.ini"


@dataclass(frozen=True)
class AppConfig:
    # [server]
    server_host: str = "0.0.0.0"
    server_port: int = 8015

    # [auth]
    api_key: str = "CHANGE_ME_TO_SECURE_KEY"

    # [ollama]
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:20b"
    ollama_num_ctx: int = 65536
    ollama_timeout: int = 600
    ollama_temperature: float = 0.3
    ollama_num_predict: int = 4096
    ollama_repeat_penalty: float = 1.3
    ollama_repeat_last_n: int = 256

    # [ocr]
    ocr_languages: list[str] = ("uk", "en")
    ocr_use_gpu: bool = False

    # [processing]
    max_upload_size_mb: int = 50
    max_total_tokens_estimate: int = 60000
    prompt_file: str = "prompts/analysis_prompt.txt"
    extraction_prompt_file: str = "prompts/extraction_prompt.txt"
    report_prompt_file: str = "prompts/report_prompt.txt"


def _load_config(path: Path) -> AppConfig:
    """Parse config.ini and return AppConfig with defaults for missing values."""
    parser = configparser.ConfigParser()

    if path.exists():
        parser.read(path, encoding="utf-8")
        logger.info("Configuration loaded from %s", path)
    else:
        logger.warning("Config file not found at %s, using defaults", path)
        return AppConfig()

    def get(section: str, key: str, fallback: str | None = None) -> str | None:
        return parser.get(section, key, fallback=fallback)

    def getint(section: str, key: str, fallback: int = 0) -> int:
        return parser.getint(section, key, fallback=fallback)

    def getfloat(section: str, key: str, fallback: float = 0.0) -> float:
        return parser.getfloat(section, key, fallback=fallback)

    def getbool(section: str, key: str, fallback: bool = False) -> bool:
        return parser.getboolean(section, key, fallback=fallback)

    languages_raw = get("ocr", "languages", "uk,en")
    languages = [lang.strip() for lang in languages_raw.split(",")]

    return AppConfig(
        server_host=get("server", "host", "0.0.0.0"),
        server_port=getint("server", "port", 8015),
        api_key=get("auth", "api_key", "CHANGE_ME_TO_SECURE_KEY"),
        ollama_base_url=get("ollama", "base_url", "http://localhost:11434"),
        ollama_model=get("ollama", "model", "gpt-oss:20b"),
        ollama_num_ctx=getint("ollama", "num_ctx", 65536),
        ollama_timeout=getint("ollama", "timeout_seconds", 600),
        ollama_temperature=getfloat("ollama", "temperature", 0.3),
        ollama_num_predict=getint("ollama", "num_predict", 4096),
        ollama_repeat_penalty=getfloat("ollama", "repeat_penalty", 1.3),
        ollama_repeat_last_n=getint("ollama", "repeat_last_n", 256),
        ocr_languages=languages,
        ocr_use_gpu=getbool("ocr", "use_gpu", False),
        max_upload_size_mb=getint("processing", "max_upload_size_mb", 50),
        max_total_tokens_estimate=getint("processing", "max_total_tokens_estimate", 60000),
        prompt_file=get("processing", "prompt_file", "prompts/analysis_prompt.txt"),
        extraction_prompt_file=get("processing", "extraction_prompt_file", "prompts/extraction_prompt.txt"),
        report_prompt_file=get("processing", "report_prompt_file", "prompts/report_prompt.txt"),
    )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return cached application config (singleton)."""
    return _load_config(CONFIG_FILE)
