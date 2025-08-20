# 🐦 Twitter/X Influencer Knowledge Base (RAG Assistant)

![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112+-009688?logo=fastapi&logoColor=white)
![Pytest](https://img.shields.io/badge/tests-passing-brightgreen?logo=pytest)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

An AI-powered Retrieval-Augmented Generation (RAG) system that helps users discover and query information about influencers across domains such as AI, fitness, finance, fashion, gaming, and memes. It ingests influencer data, builds a vector database, and answers questions with citations so responses are both accurate and attributable.

---

## 🔗 Quick Links
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quickstart](#-quickstart)
- [Configuration](#-configuration-env)
- [API Endpoints](#-api-endpoints)
- [ETL and Persistence](#-etl-and-persistence)
- [Developer CLI & Commands](#-developer-cli-and-commands)
- [Docker & Compose](#-docker--compose)
- [Testing](#-testing)
- [Tech Notes](#-tech-notes)

---

## ✅ Features
- 🔍 Data Ingestion Pipeline: collect and process influencer data (profiles, niches, posts)
- 🧠 Vector Search: embeddings with OpenAI (if available) and FAISS or NumPy fallback
- 🤖 RAG Engine: combines vector retrieval with LLM generation; deterministic offline fallback
- 📜 Citations: responses reference influencer names and handles
- 🌐 API: FastAPI-powered REST endpoints
- 🧪 Testing: Pytest coverage for RAG, vector search, API, persistence, and fallbacks
- 📦 Infra: Docker + docker-compose for easy deployment and persistence

---

## 🏗 Project Structure
```text
twitter-influencer-assistant/
  app/
    __init__.py
    api.py          # FastAPI app: /healthz, /ingest, /query, /feedback; mounts /mock routes
    routes.py       # Mock/demo routes (mounted under /mock)
    config.py       # Centralized configs (paths, tunables, dotenv)
    embeddings.py   # Embedding + VectorStore (FAISS/NumPy, save/load persistence)
    pipeline.py     # ETL CLI: raw -> processed with normalization, dedupe, chunking
    rag.py          # RAG orchestration with OpenAI + offline fallback
  data/
    raw/            # raw influencer data (JSON/CSV)
    processed/      # cleaned dataset produced by ETL
  infra/
    Dockerfile
    docker-compose.yml
    requirements.txt
  models/           # persisted vector store (created after first /ingest)
  tests/
    conftest.py
    test_api.py
    test_rag.py
    test_vector.py
  COMMANDS.txt      # handy commands for setup, run, tests, ETL, Docker
  pyproject.toml
  README.md
```

---

## 🧭 Architecture (High-Level)
```mermaid
graph TD
    A[Raw JSON/CSV in data/raw] --> B[ETL: app/pipeline.py\n-> data/processed/processed.json]
    B --> C[VectorStore (FAISS/NumPy)]
    C -->|persist| D[models/vector_store/]
    E[FastAPI app.api] --> C
    E --> F[/ingest, /query, /feedback, /healthz]
    E --> G[/mock/query (demo)]
```

---

## 🚀 Quickstart
1) Install dependencies
```bash
pip install -r infra/requirements.txt
```

2) Run the API locally (dev)
```bash
uvicorn app.api:app --reload
```
API will start at `http://127.0.0.1:8000`.

3) Run tests
```bash
pytest -q
```

---

## ⚙ Configuration (.env)
Create a `.env` file in the project root (dotenv is loaded automatically):

```ini
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Vector search
VECTOR_TOP_K=3
USE_FAISS=auto   # values: auto (default), true/1, false/0 (force NumPy fallback)
MAX_CHUNK_LEN=280
```

---

## 🌐 API Endpoints

| Method | Path         | Description                                      |
|--------|--------------|--------------------------------------------------|
| GET    | /healthz     | Health check                                     |
| POST   | /ingest      | Ingest dataset JSON (path in request body)       |
| POST   | /query       | Ask a question; returns answer + citations       |
| POST   | /feedback    | Submit feedback on a query result                |
| POST   | /mock/query  | Demo endpoint using static mock docs (no vector) |

Example requests:
```bash
# Health check
curl http://127.0.0.1:8000/healthz

# Ingest & Query
curl -X POST http://127.0.0.1:8000/ingest -H 'Content-Type: application/json' -d '{"dataset_path":"data/raw/sample.json"}'
curl -X POST http://127.0.0.1:8000/query  -H 'Content-Type: application/json' -d '{"query":"AI startups"}'
```

---

## 🧱 ETL and Persistence

### ETL: raw -> processed
Run ETL over a directory or a single file to produce `data/processed/processed.json`:
```bash
# Directory (default)
python -m app.pipeline --input data/raw --output-file data/processed/processed.json --max-chunk-len 280

# Single file
python -m app.pipeline --input data/raw/sample.json --output-file data/processed/processed.json
```
What ETL does:
- Loads JSON/CSV
- Cleans and normalizes fields (names, handles, niches), dedupes, validates
- Optionally chunks long `sample_post` fields

### Vector store persistence
- After the first `/ingest`, the vector store is saved under `models/vector_store/` (metadata + FAISS/NumPy data)
- On app startup, it attempts to load the persisted store automatically so `/query` works without re-ingesting

---

## 🛠 Developer CLI and Commands
Most common commands are listed in `COMMANDS.txt` at the repo root.

Key examples:
```bash
# Dev server
uvicorn app.api:app --reload

# Tests
pytest -q
```

---

## 🐳 Docker & Compose

### Docker (single container)
```bash
docker build -t twitter-assistant .
docker run -p 8000:8000 twitter-assistant
```

### Docker Compose (recommended)
```bash
docker-compose -f infra/docker-compose.yml up --build
```
Notes:
- The compose file mounts `./models` -> `/app/models` to persist the vector store across container restarts
- A healthcheck probes `GET /healthz`

---

## ✅ Testing
Run the full test suite:
```bash
pytest -q
```
Included coverage:
- `/ingest`, `/query`, `/feedback`, `/healthz`
- Vector retrieval and keyword fallback
- Persistence (save/load) and OpenAI-offline fallback

---

## 🧩 Tech Notes
- FAISS usage is configurable via `USE_FAISS` (auto/true/false)
- Embeddings use OpenAI when `OPENAI_API_KEY` is set; otherwise a deterministic hashed fallback is used

---

## 📜 License
TBD
