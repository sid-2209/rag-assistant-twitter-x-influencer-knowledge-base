from __future__ import annotations

from pathlib import Path
from typing import Final
import os

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Base directories
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
MODELS_DIR: Final[Path] = BASE_DIR / "models"

# Models and pipeline defaults (placeholders)
DEFAULT_EMBEDDING_MODEL_NAME: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL_NAME: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_GENERATION_MODEL_NAME: Final[str] = "gpt-4o-mini"  # Placeholder; replace with your LLM provider

# API and app constants
APP_NAME: Final[str] = "twitter-influencer-assistant"
API_VERSION: Final[str] = "0.1.0"

# Vector store and pipeline tunables
VECTOR_TOP_K: Final[int] = int(os.getenv("VECTOR_TOP_K", "3"))
VECTOR_PERSIST_SUBDIR: Final[str] = os.getenv("VECTOR_PERSIST_SUBDIR", "vector_store")
VECTOR_PERSIST_DIR: Final[Path] = MODELS_DIR / VECTOR_PERSIST_SUBDIR
DEFAULT_MAX_CHUNK_LEN: Final[int] = int(os.getenv("MAX_CHUNK_LEN", "280"))

# FAISS toggle: "auto" (default), "true"/"1" to enable if available, "false"/"0" to force disable
FAISS_MODE: Final[str] = os.getenv("USE_FAISS", "auto").strip().lower()

# Create directories if they don't exist (safe for first run)
for directory in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
