from __future__ import annotations

from typing import List, Literal, Any, Dict

from fastapi import FastAPI, HTTPException
from app import routes
from pydantic import BaseModel, Field

from .embeddings import VectorStore
from .rag import generate_answer
import json
from pathlib import Path

# In-memory database stub populated via /ingest
db_stub: List[Dict[str, Any]] = []

# Attempt to load persisted vector store; fallback to empty store
from .config import MODELS_DIR, VECTOR_TOP_K, VECTOR_PERSIST_DIR  # late import to avoid cycles

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


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]


class FeedbackRequest(BaseModel):
    query_id: str
    rating: Literal["up", "down"]


class FeedbackResponse(BaseModel):
    status: str


app = FastAPI(title="Twitter Influencer Assistant")


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


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if not db_stub:
        return QueryResponse(answer="No data ingested yet", citations=[])

    results = [
        r for r in vector_store.search(request.query, top_k=VECTOR_TOP_K)
        if float(r.get("score", 0.0)) > 0.0
    ]
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

    rag_result = generate_answer(request.query, results)
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
