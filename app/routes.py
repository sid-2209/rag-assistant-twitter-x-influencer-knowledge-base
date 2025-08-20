from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List

from app.rag import generate_answer

router = APIRouter()


# ---- Request/Response Models ----
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]


# ---- Placeholder: mock influencer docs ----
# Later this will be replaced by real vector DB retrieval
MOCK_DOCS = [
    {
        "name": "Alex Johnson",
        "handle": "@alexj",
        "niche": "Tech & AI",
        "sample_post": "Explaining RAG systems with memes 🤖🔥",
    },
    {
        "name": "Sophie Lee",
        "handle": "@sophie_lee",
        "niche": "Lifestyle & Productivity",
        "sample_post": "5 hacks to balance work & content creation ✨",
    },
]


@router.post("/query", response_model=QueryResponse)
async def query_influencers(request: QueryRequest) -> QueryResponse:
    try:
        # 🔹 for now, always use MOCK_DOCS
        result = generate_answer(request.query, MOCK_DOCS)
        return QueryResponse(answer=result["answer"], citations=result["citations"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")