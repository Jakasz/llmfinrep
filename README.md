# Financial Statement Analyzer

> AI-powered service for automated financial analysis of corporate documents using a hybrid LLM + Python pipeline.

## Overview

A FastAPI microservice that accepts financial documents (PDF, images, Excel, Word), extracts structured data using a local LLM, computes **financial indicators** with deterministic Python logic, and generates professional HTML reports — all without sending data to external APIs.

### Why Hybrid Architecture?

Local LLMs (~20B parameters) excel at document parsing and natural language generation, but struggle with precise arithmetic in complex financial formulas. This service solves the problem with a **3-step hybrid pipeline**:

```
Documents → Text Extraction → LLM (parse → JSON) → Python (calculate) → LLM (generate HTML report)
```

- **Step 1** — LLM extracts financial row values from unstructured text into structured JSON
- **Step 2** — Python calculates all financial ratios with guaranteed arithmetic precision
- **Step 3** — LLM formats pre-calculated results into a professional HTML report

This approach combines the flexibility of LLMs with 100% calculation accuracy.

## Financial Analysis

The service computes **financial indicators** across **6 analysis blocks**, based on Ukrainian financial statement forms (Balance Sheet — Form №1, Income Statement — Form №2):

| Block | Indicators | Examples |
|-------|-----------|----------|
| **Liquidity** | 3 | Current ratio, Quick ratio, Absolute liquidity |
| **Financial Stability** | 2 | Autonomy coefficient, Financial leverage |
| **Business Activity** | 6 | Asset turnover, Inventory turnover, AR/AP turnover |
| **Cash Flow** | 5 | Operating CF, Investing CF, Free CF |
| **Balance Structure** | 7 | Asset composition, Depreciation, Asset quality |
| **Profitability** | 6 | ROA, ROE, ROCE, ROS, ROCA, RONCA |

Each indicator includes the formula with actual values, the computed result, and a color-coded rating (green / orange / red).

## Tech Stack

- **FastAPI** + Uvicorn — async web framework
- **Ollama** — local LLM inference (privacy-first, no cloud calls)
- **PaddleOCR** — document image recognition (Ukrainian + English)
- **PyMuPDF** — PDF text extraction
- **openpyxl** / **python-docx** — Excel and Word support

## Supported File Types

| Format | Engine |
|--------|--------|
| PDF (native text) | PyMuPDF |
| PDF (scanned) | PyMuPDF + PaddleOCR |
| PNG / JPG / JPEG | PaddleOCR |
| XLSX | openpyxl |
| DOCX | python-docx |

## API

### `GET /health`
Health check — returns service status and Ollama connectivity. No authentication required.

### `POST /api/v1/analyze`
Upload up to 10 financial documents for analysis. Requires Bearer token authentication.

**Request:** `multipart/form-data` with `files` field

**Response:**
```json
{
  "report_html": "<html>...</html>",
  "extracted_data": {
    "company_name": "...",
    "period": "...",
    "balance_start": {},
    "balance_end": {},
    "income_current": {}
  },
  "timing": {
    "extraction_llm_sec": 12.3,
    "calculation_sec": 0.01,
    "report_llm_sec": 8.7,
    "total_sec": 21.0
  }
}
```

## Project Structure

```
├── app/
│   ├── main.py                 # FastAPI app, CORS, lifespan
│   ├── config.py               # Configuration reader
│   ├── auth.py                 # Bearer token authentication
│   ├── routers/
│   │   └── analyze.py          # /api/v1/analyze endpoint (3-step pipeline)
│   ├── services/
│   │   ├── extractor.py        # File type dispatcher
│   │   ├── pdf_extractor.py    # PDF text extraction + OCR
│   │   ├── image_extractor.py  # Image OCR (PaddleOCR v5)
│   │   ├── excel_extractor.py  # Excel extraction
│   │   ├── docx_extractor.py   # Word extraction
│   │   ├── llm_client.py       # Ollama API client
│   │   ├── json_extractor.py   # JSON parsing & validation
│   │   └── financial_calculator.py  # financial ratios (Python)
│   └── utils/
│       └── text_utils.py       # Text cleanup & token estimation
├── prompts/                    # LLM prompt templates
├── config.ini                  # Application configuration
├── requirements.txt            # Python dependencies
└── run.py                      # Entry point
```

## Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) with a compatible model

### Installation

```bash
git clone https://github.com/Jakasz/llmfinrep.git
cd llmfinrep
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Configuration

Copy and edit the configuration file:

```bash
cp config.ini.example config.ini
```

Key settings in `config.ini`:
- `auth.api_key` — Bearer token for API access
- `ollama.base_url` — Ollama server address
- `ollama.model` — LLM model name
- `ocr.use_gpu` — enable GPU acceleration for OCR

### Run

```bash
python run.py
```

The service starts at `http://localhost:8015` by default.

## License

This project is part of a personal portfolio. All rights reserved.
