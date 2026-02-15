"""FastAPI application: lifespan, middleware, and route registration."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_config
from app.routers.analyze import router as analyze_router
from app.services.image_extractor import init_ocr_engine
from app.services.llm_client import check_ollama_health

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize resources at startup, clean up on shutdown."""
    config = get_config()
    logger.info("Starting Counterparty Financial Analyzer service")
    logger.info("Server: %s:%d", config.server_host, config.server_port)
    logger.info("Ollama: %s (model: %s)", config.ollama_base_url, config.ollama_model)

    # Ініціалізація OCR-движка при старті (завантаження моделі)
    try:
        init_ocr_engine(config)
    except Exception as e:
        logger.error("Failed to initialize OCR engine: %s", e, exc_info=True)
        logger.warning("OCR features will not be available. Image and scanned PDF processing will fail.")

    # Перевірка з'єднання з Ollama
    ollama_status = await check_ollama_health(config)
    if ollama_status["ollama_reachable"]:
        logger.info("Ollama is reachable")
        if ollama_status["model_available"]:
            logger.info("Model '%s' is available", config.ollama_model)
        else:
            logger.warning(
                "Model '%s' not found. Available models: %s",
                config.ollama_model,
                ollama_status.get("available_models", []),
            )
    else:
        logger.warning("Ollama is not reachable: %s", ollama_status.get("error", "unknown"))

    yield

    logger.info("Shutting down Counterparty Financial Analyzer service")


app = FastAPI(
    title="Counterparty Financial Analyzer",
    description="Service for analyzing counterparty documents using LLM",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(analyze_router)


@app.get("/health")
async def health_check():
    """Health check endpoint — returns service status and Ollama connectivity."""
    config = get_config()
    ollama_status = await check_ollama_health(config)
    return {
        "status": "ok",
        "service": "counterparty-financial-analyzer",
        "version": "1.0.0",
        "ollama": ollama_status,
    }
