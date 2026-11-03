import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
LOG_PATH = Path(os.getenv("RAG_LOG_PATH", ROOT / "logs" / "requests.jsonl"))

CHUNK_SIZE = 400          # characters per chunk
CHUNK_OVERLAP = 60
TOP_K = 3
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.12"))  # below this -> refuse
MIN_COVERAGE = float(os.getenv("MIN_COVERAGE", "0.5"))  # share of question terms the context must explain

# LLM provider: "extractive" (offline, deterministic) or "openai" (any OpenAI-compatible API)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "extractive")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# illustrative pricing, USD per 1K tokens (input, output)
PRICE_IN, PRICE_OUT = 0.00015, 0.0006
