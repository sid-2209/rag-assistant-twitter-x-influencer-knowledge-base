from __future__ import annotations

from typing import List, Literal, Any, Dict

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from app import routes
from pydantic import BaseModel, Field

from .embeddings import VectorStore
from .rag import generate_answer
from .rag_langchain import generate_answer_langchain
import os
from fastapi.staticfiles import StaticFiles
import json
from pathlib import Path
from fastapi.responses import FileResponse

# In-memory database stub populated via /ingest
db_stub: List[Dict[str, Any]] = []

# Attempt to load persisted vector store; fallback to empty store
from .config import (
    MODELS_DIR,
    VECTOR_TOP_K,
    VECTOR_PERSIST_DIR,
    DEFAULT_MAX_CHUNK_LEN,
    RERANKER_ENABLED,
)

persist_dir = VECTOR_PERSIST_DIR
_loaded = VectorStore.load(persist_dir)
vector_store = _loaded if _loaded is not None else VectorStore()


class IngestRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to the dataset to ingest")


class IngestResponse(BaseModel):
    status: str
    count: int | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query text")
    implementation: Literal["vanilla", "langchain"] | None = Field("vanilla")
    model: str | None = Field(None, description="Model name, e.g., gpt-4o-mini")
    api_key: str | None = Field(None, description="Optional API key to use at runtime")
    base_url: str | None = Field(None, description="Optional OpenAI-compatible base URL (e.g., Groq or Together)")


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]


class FeedbackRequest(BaseModel):
    query_id: str
    rating: Literal["up", "down"]


class FeedbackResponse(BaseModel):
    status: str


app = FastAPI(title="Twitter Influencer Assistant")

# Minimal HTML UI (optional)
if os.getenv("ENABLE_WEB_UI", "false").lower() in ("1", "true", "yes", "on"):
    from .webui import router as ui_router  # delayed import to avoid cycles
    app.include_router(ui_router)

# Serve static assets (e.g., lottie background) - AFTER router inclusion
static_dir = str(Path(__file__).resolve().parent / "static")
print(f"Mounting static files from: {static_dir}")
print(f"Static directory exists: {Path(static_dir).exists()}")
if Path(static_dir).exists():
    print(f"Static directory contents: {list(Path(static_dir).rglob('*'))}")

try:
    app.mount(
        "/static",
        StaticFiles(directory=static_dir),
        name="static",
    )
    print("Static files mounted successfully")
except Exception as e:
    print(f"Failed to mount static files: {e}")


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    global db_stub, vector_store
    dataset_path = Path(request.dataset_path)
    # Allow relative paths from project root
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parent.parent / dataset_path

    # Prefer processed dataset via ETL if input is a directory
    data: Any
    if dataset_path.is_dir():
        from .pipeline import run_pipeline  # lazy import

        processed_path = Path(__file__).resolve().parent.parent / "data/processed/processed.json"
        try:
            run_pipeline(dataset_path, processed_path, DEFAULT_MAX_CHUNK_LEN)
            with processed_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ETL failed: {e}")
    else:
        try:
            with dataset_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read dataset: {e}")
    # Expecting a list of dicts
    if isinstance(data, list):
        db_stub = [dict(record) for record in data]
        # Build documents for vector store: concatenate niche + sample_post
        docs: List[Dict[str, Any]] = []
        for record in db_stub:
            text_blob = f"{record.get('niche', '')}. {record.get('sample_post', '')}"
            docs.append(
                {
                    "text": text_blob,
                    "metadata": {
                        "id": record.get("id"),
                        "name": record.get("name"),
                        "handle": record.get("handle"),
                        "niche": record.get("niche"),
                        "followers": record.get("followers"),
                        "sample_post": record.get("sample_post"),
                    },
                }
            )
        # Reset vector store for idempotent test runs
        vector_store = VectorStore()
        vector_store.add_documents(docs)
        try:
            vector_store.save(persist_dir)
        except Exception:
            pass
        return IngestResponse(status="ingested", count=len(db_stub))
    # Fallback if file is not a list
    db_stub = []
    vector_store = VectorStore()
    return IngestResponse(status="ingested", count=0)
@app.post("/upload_dataset", response_model=IngestResponse)
async def upload_dataset(
    file: UploadFile = File(...),
):
    """Upload a dataset file; save to data/raw and run ETL via ingest flow."""
    raw_dir = Path(__file__).resolve().parent.parent / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = raw_dir / file.filename
    content = await file.read()
    target_path.write_bytes(content)
    # Reuse ingest with directory path to trigger ETL
    return await ingest(IngestRequest(dataset_path=str(raw_dir)))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if not db_stub:
        return QueryResponse(answer="No data ingested yet", citations=[])

    results = [
        r for r in vector_store.search(request.query, top_k=VECTOR_TOP_K)
        if float(r.get("score", 0.0)) > 0.0
    ]

    # Optional lightweight reranker by keyword overlap
    if RERANKER_ENABLED and results:
        q_tokens = {t for t in request.query.lower().split() if t.isalnum()}
        def overlap(meta: Dict[str, Any]) -> int:
            text = f"{meta.get('niche','')} {meta.get('sample_post','')} {meta.get('name','')} {meta.get('handle','')}".lower()
            return sum(1 for t in q_tokens if t in text)
        results.sort(key=overlap, reverse=True)
    if not results:
        # Fallback: simple keyword match on ingested records
        q = request.query.lower()
        keyword_hits: List[Dict[str, Any]] = []
        for record in db_stub:
            haystacks = [
                str(record.get("niche", "")),
                str(record.get("sample_post", "")),
                str(record.get("name", "")),
                str(record.get("handle", "")),
            ]
            if any(q in h.lower() for h in haystacks):
                keyword_hits.append(record)

        if not keyword_hits:
            return QueryResponse(answer="No influencers found", citations=[])

        rag_result = generate_answer(request.query, keyword_hits)
        citations = [
            {k: c.get(k) for k in ("name", "handle", "niche", "followers")}
            for c in rag_result.get("citations", [])
        ]
        return QueryResponse(answer=rag_result.get("answer", ""), citations=citations)

    # Allow runtime model/key for future extensions
    implementation = (request.implementation or "vanilla").lower()
    if implementation == "langchain":
        rag_result = generate_answer_langchain(request.query, results, request.model or "gpt-4o-mini", request.api_key)
    else:
        rag_result = generate_answer(
            request.query,
            results,
            model=request.model or None,
            api_key=request.api_key or None,
            base_url=request.base_url or None,
        )
    citations = [
        {k: c.get(k) for k in ("name", "handle", "niche", "followers")}
        for c in rag_result.get("citations", [])
    ]
    return QueryResponse(answer=rag_result.get("answer", ""), citations=citations)


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    _ = request  # Placeholder: store feedback for RLHF or analytics later
    return FeedbackResponse(status="feedback recorded")

# include endpoints from routes.py under a distinct prefix to avoid overriding /query
app.include_router(routes.router, prefix="/mock")


@app.get("/debug/static")
async def debug_static() -> Dict[str, Any]:
    """Debug endpoint to check static file configuration"""
    static_dir = Path(__file__).resolve().parent / "static"
    return {
        "static_dir": str(static_dir),
        "exists": static_dir.exists(),
        "contents": list(static_dir.rglob("*")) if static_dir.exists() else [],
        "lottie_exists": (static_dir / "lottie" / "bg.json").exists(),
        "icon_exists": (static_dir / "icons" / "RAG icon.png").exists(),
    }

@app.get("/static/{path:path}")
async def serve_static_fallback(path: str):
    """Fallback static file serving"""
    static_dir = Path(__file__).resolve().parent / "static"
    file_path = static_dir / path
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine content type based on file extension
    content_type = "text/plain"
    if path.endswith(".json"):
        content_type = "application/json"
    elif path.endswith(".png"):
        content_type = "image/png"
    elif path.endswith(".jpg") or path.endswith(".jpeg"):
        content_type = "image/jpeg"
    elif path.endswith(".css"):
        content_type = "text/css"
    elif path.endswith(".js"):
        content_type = "application/javascript"
    
    return FileResponse(file_path, media_type=content_type)
