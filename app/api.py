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
    hallucination_analysis: Dict[str, Any]


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

# Debug static file configuration
static_dir = str(Path(__file__).resolve().parent / "static")
print(f"Static files directory: {static_dir}")
print(f"Static directory exists: {Path(static_dir).exists()}")

# Note: Static files are served via the /static/{path:path} endpoint below


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
    global vector_store
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        print(f"Processing query: {query_text}")
        
        # Search for relevant documents
        results = vector_store.search(query_text, VECTOR_TOP_K)
        print(f"Found {len(results)} results from vector search")
        
        # Filter results with positive scores
        filtered_results = [r for r in results if r.get("score", 0) > 0]
        print(f"Filtered to {len(filtered_results)} results with positive scores")
        
        # Intelligent filtering for handle-specific queries
        import re
        handle_match = re.search(r'@(\w+)', query_text)
        if handle_match:
            target_handle = handle_match.group(1).lower()
            print(f"Detected handle-specific query for: @{target_handle}")
            
            # Check if query is asking about handle existence vs handle content
            query_lower = query_text.lower()
            is_existence_query = any(word in query_lower for word in ["exist", "exists", "mentioned", "found", "appear", "mention"])
            is_content_query = any(word in query_lower for word in ["niche", "who", "what", "content", "post", "tweet", "followers"])
            
            # If it's a content query about a specific handle, filter for mentions
            if is_content_query and target_handle:
                is_existence_query = True  # Treat content queries as existence queries for filtering
            
            if is_existence_query:
                # Filter to only include citations that mention the handle
                handle_mentioned_results = []
                for result in filtered_results:
                    # Check in the content/sample_post
                    content = result.get("sample_post", "") or result.get("text", "") or result.get("content", "")
                    if target_handle in content.lower():
                        handle_mentioned_results.append(result)
                
                if handle_mentioned_results:
                    print(f"Filtered to {len(handle_mentioned_results)} results mentioning handle @{target_handle}")
                    filtered_results = handle_mentioned_results
                else:
                    print(f"No results found mentioning handle @{target_handle}")
                    # Keep original results but mark that the specific handle wasn't found
                    filtered_results = filtered_results[:3]  # Limit to top 3 for context
            else:
                # Filter to only include citations from the specific handle
                handle_specific_results = []
                for result in filtered_results:
                    metadata = result.get("metadata", {})
                    handle = metadata.get("handle", result.get("handle", ""))
                    if handle and target_handle in handle.lower():
                        handle_specific_results.append(result)
                
                if handle_specific_results:
                    print(f"Filtered to {len(handle_specific_results)} results from handle @{target_handle}")
                    filtered_results = handle_specific_results
                else:
                    print(f"No results found for handle @{target_handle}")
                    # Keep original results but mark that the specific handle wasn't found
                    filtered_results = filtered_results[:3]  # Limit to top 3 for context
        
        if not filtered_results:
            print("No filtered results found, returning default response")
            return QueryResponse(
                answer="I don't have enough information to answer this question. Please upload relevant data first.",
                citations=[],
                hallucination_analysis={
                    "is_hallucination": True,
                    "confidence": "high",
                    "score": 1.0,
                    "reason": "No relevant citations found",
                    "suggestions": ["Upload more relevant data", "Try a different query"]
                }
            )
        
        print("Generating answer with hallucination detection...")
        # Generate answer with hallucination detection
        response = generate_answer(
            query=query_text,
            citations=filtered_results,
            model=request.model,
            api_key=request.api_key,
            base_url=request.base_url
        )
        
        # Ensure response is valid and contains meaningful content
        if not response or not isinstance(response, dict):
            print("Invalid response structure, creating fallback")
            response = {
                "answer": "I couldn't generate a response. Please try again.",
                "citations": filtered_results,
                "hallucination_analysis": {
                    "is_hallucination": True,
                    "confidence": "high", 
                    "score": 1.0,
                    "reason": "Response generation failed",
                    "suggestions": ["Try again with a different query"]
                }
            }

        answer = response.get('answer', '') or ''
        
        # Check if the answer is meaningful (not an error message)
        error_indicators = [
            "i couldn't generate a response",
            "i apologize, but",
            "no response generated",
            "error occurred",
            "failed to generate",
            "api error",
            "model error"
        ]
        
        # More specific check to avoid triggering on legitimate "not found" responses
        is_error_answer = any(indicator in answer.lower() for indicator in error_indicators) and not (
            "couldn't find" in answer.lower() or 
            "not found" in answer.lower() or
            "no information" in answer.lower()
        )
        
        if is_error_answer and filtered_results:
            print("Detected error answer, generating fallback from citations")
            # Generate a fallback answer directly from citations
            from .rag import _generate_fallback_answer
            fallback_answer = _generate_fallback_answer(query_text, filtered_results)
            response["answer"] = fallback_answer
            answer = fallback_answer
        
        print(f"Generated response with answer length: {len(answer)}")
        print(f"Hallucination analysis: {response.get('hallucination_analysis', {}).get('is_hallucination', 'unknown')}")
        
        return QueryResponse(
            answer=response["answer"],
            citations=response["citations"],
            hallucination_analysis=response["hallucination_analysis"]
        )
        
    except Exception as e:
        print(f"Error in query endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    _ = request  # Placeholder: store feedback for RLHF or analytics later
    return FeedbackResponse(status="feedback recorded")

# include endpoints from routes.py under a distinct prefix to avoid overriding /query
app.include_router(routes.router, prefix="/mock")


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
