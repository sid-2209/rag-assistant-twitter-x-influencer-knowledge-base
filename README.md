# 🐦 Twitter/X Influencer Knowledge Base (RAG Assistant)

![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112+-009688?logo=fastapi&logoColor=white)
![Pytest](https://img.shields.io/badge/tests-passing-brightgreen?logo=pytest)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

An AI-powered Retrieval-Augmented Generation (RAG) system that helps users discover and query information about Twitter/X influencers. It ingests influencer datasets, builds a vector database, and answers questions with citations so responses are both accurate and attributable.

---

## 🖥️ Demo

### Homepage
![Homepage](app/static/demo_images/homepage.png)

### History Page
![History](app/static/demo_images/History.png)

---

## ✅ Features
- 🔍 **Data Ingestion Pipeline**: Process influencer data (JSON/CSV) with ETL pipeline (cleaning, normalization, deduplication)
- 🧠 **Vector Search**: OpenAI embeddings + free local sentence-transformers fallback + hashed fallback
- 🤖 **RAG Engine**: Vanilla pipeline + LangChain-style mode (toggle at query-time)
- 🧰 **Flexible Models**: OpenAI, Groq, Together AI, and other OpenAI-compatible providers via custom base URL
- 📜 **Citations**: Responses reference influencer names and handles
- 🌐 **API**: FastAPI endpoints for health, ingest, upload, query, feedback
- 🖥 **Web UI**: Modern HTML interface with glassmorphism design, file upload, query interface, and help tooltips
- 🧪 **Testing**: Pytest coverage for RAG, vector search, API, persistence, and fallbacks
- 📦 **Infra**: Docker + docker-compose + GitHub Actions CI (build/push image)

---

## 🏗 Project Structure
```text
twitter-influencer-assistant/
  app/
    __init__.py
    api.py              # FastAPI app: /healthz, /ingest, /upload_dataset, /query, /feedback
    routes.py           # Mock/demo routes (mounted under /mock)
    config.py           # Centralized configs (paths, tunables, dotenv)
    embeddings.py       # Embedding + VectorStore (FAISS/NumPy, ChromaDB, save/load persistence)
    pipeline.py         # ETL CLI: raw -> processed with normalization, dedupe, chunking
    rag.py              # RAG orchestration with OpenAI + offline fallback
    rag_langchain.py    # LangChain-style RAG (lightweight placeholder)
    webui.py            # Web UI router (/ui, /ui/upload, /ui/query, /ui/about)
    templates/
      index.html        # Main UI template with glassmorphism design
      about.html        # About page template
    static/
      icons/            # UI icons (RAG icon.png)
      demo_images/      # Demo screenshots
      lottie/           # Background animations (bg.json)
  data/
    raw/                # Raw influencer data (JSON/CSV)
    processed/          # Cleaned dataset produced by ETL
  infra/
    Dockerfile
    docker-compose.yml
    requirements.txt
  models/               # Persisted vector store (created after first /ingest)
  tests/
    conftest.py
    test_api.py
    test_rag.py
    test_vector.py
  COMMANDS.txt          # Handy commands for setup, run, tests, ETL, Docker
  pyproject.toml
  README.md
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

3) Enable Web UI (optional)
```bash
ENABLE_WEB_UI=true uvicorn app.api:app --reload
```
Web UI will be available at `http://127.0.0.1:8000/ui`.

4) Run tests
```bash
pytest -q
```

---

## ⚙ Configuration (.env)
Create a `.env` file in the project root (dotenv is loaded automatically):

```ini
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1

# Vector search
VECTOR_TOP_K=3
USE_FAISS=auto   # values: auto (default), true/1, false/0 (force NumPy fallback)
VECTOR_BACKEND=faiss  # values: faiss (default), chroma
MAX_CHUNK_LEN=280

# Web UI
ENABLE_WEB_UI=true  # Enable HTML web interface
```

---

## 🌐 API Endpoints

| Method | Path         | Description                                      |
|--------|--------------|--------------------------------------------------|
| GET    | /healthz     | Health check                                     |
| POST   | /ingest      | Ingest dataset by path (runs ETL for directories)|
| POST   | /upload_dataset | Multipart upload (saves to data/raw and runs ETL)|
| POST   | /query       | Ask a question; returns answer + citations       |
| POST   | /feedback    | Submit feedback on a query result                |
| POST   | /mock/query  | Demo endpoint using static mock docs (no vector) |

Example requests:
```bash
# Health check
curl http://127.0.0.1:8000/healthz

# Ingest & Query
curl -X POST http://127.0.0.1:8000/ingest -H 'Content-Type: application/json' -d '{"dataset_path":"data/raw/sample.json"}'
curl -X POST http://127.0.0.1:8000/query  -H 'Content-Type: application/json' -d '{"query":"AI startups", "model":"gpt-4o-mini", "api_key":"sk-...", "base_url":"https://api.groq.com/openai/v1"}'
```

---

## 🖥 Web UI

The project includes a modern HTML web interface built with FastAPI + Jinja2 templates.

### Features
- **Glassmorphism Design**: Translucent cards with backdrop blur effects
- **File Upload**: Drag-and-drop or click to upload JSON/CSV datasets
- **Query Interface**: Natural language questions with model selection
- **Provider Support**: OpenAI, Groq, Together AI via custom base URL
- **Help Tooltips**: Contextual help icons with detailed explanations
- **Responsive Layout**: Works on desktop and mobile devices

### Access
- **Homepage**: `http://127.0.0.1:8000/ui`
- **About Page**: `http://127.0.0.1:8000/ui/about`

### Usage
1. Upload a JSON/CSV file of influencer data
2. Wait for ingestion to complete (ETL pipeline runs automatically)
3. Ask questions using natural language
4. Select model and provider (optional API key/base URL)
5. View answers with citations

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

**What ETL does:**
- Loads JSON/CSV files
- Cleans and normalizes fields (names, handles, niches)
- Deduplicates records
- Optionally chunks long `sample_post` fields
- Handles flexible field mapping for common column names

### Vector store persistence
- After the first `/ingest`, the vector store is saved under `models/vector_store/` (metadata + FAISS/NumPy data)
- On app startup, it attempts to load the persisted store automatically so `/query` works without re-ingesting
- Optional: set `VECTOR_BACKEND=chroma` to use a ChromaDB persistent store in `models/chroma_store/`

---

## 🛠 Developer CLI and Commands
Most common commands are listed in `COMMANDS.txt` at the repo root.

Key examples:
```bash
# Dev server
uvicorn app.api:app --reload

# Dev server with Web UI
ENABLE_WEB_UI=true uvicorn app.api:app --reload

# Tests
pytest -q

# ETL pipeline
python -m app.pipeline --input data/raw --output-file data/processed/processed.json
```

---

## 🐳 Docker & Compose

### Docker (single container)
```bash
docker build -t twitter-assistant .
docker run -p 8000:8000 -e ENABLE_WEB_UI=true twitter-assistant
```

### Docker Compose (recommended)
```bash
docker-compose -f infra/docker-compose.yml up --build
```

**Notes:**
- The compose file mounts `./models` → `/app/models` to persist the vector store across container restarts
- A healthcheck probes `GET /healthz`
- Web UI is enabled by default in the compose setup

---

## ✅ Testing
Run the full test suite:
```bash
pytest -q
```

**Included coverage:**
- `/ingest`, `/query`, `/feedback`, `/healthz`
- Vector retrieval and keyword fallback
- Persistence (save/load) and OpenAI-offline fallback
- Local embeddings with sentence-transformers
- ChromaDB vector store integration

---

## 🧩 Tech Notes

### Embeddings
- **Primary**: OpenAI embeddings (when API key is provided)
- **Fallback 1**: Local sentence-transformers (all-MiniLM-L6-v2) - free and offline
- **Fallback 2**: Deterministic hashed embeddings - always available

### Vector Stores
- **Default**: FAISS/NumPy implementation with persistence
- **Optional**: ChromaDB backend (set `VECTOR_BACKEND=chroma`)

### Model Providers
- **OpenAI**: Default provider with GPT models
- **Groq**: Fast inference with LLaMA, Mixtral models
- **Together AI**: Alternative provider with various models
- **Custom**: Any OpenAI-compatible API via base URL

### Web UI
- **Framework**: FastAPI + Jinja2 templates
- **Styling**: Custom CSS with glassmorphism effects
- **Animations**: Lottie background animations
- **Accessibility**: Keyboard navigation and ARIA labels

---

## 🚚 CI/CD & Deployment

- **GitHub Actions CI**: Configured in `.github/workflows/ci.yml` to run tests and build/push Docker image to GHCR on pushes to `main`
- **Render**: See `infra/render.yaml` for Docker-based deployment
- **Cloud Run**: See `infra/cloudrun.md` for manual deployment steps

---

## 📜 License

MIT License

Copyright (c) 2025 Siddhartha Srivastava

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
