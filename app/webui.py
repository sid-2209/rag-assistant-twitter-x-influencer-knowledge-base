from __future__ import annotations

from fastapi import APIRouter, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os
import uuid
from typing import Dict, Any, List
from collections import Counter
from datetime import datetime


templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

router = APIRouter()

# In-memory cache of the last ingest for UI display/download
LAST_INGEST: Dict[str, Any] = {}
FIRST_N: int = 20

# In-memory history storage for all interactions
INTERACTION_HISTORY: List[Dict[str, Any]] = []


def log_interaction(
    action: str,
    dataset: str = None,
    query: str = None,
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    answer: str = None,
    citations: List[Dict[str, Any]] = None,
    hallucination_analysis: Dict[str, Any] = None,
    ingested: int = None
):
    """Log an interaction to the history."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "dataset": dataset,
        "query": query,
        "model": model,
        "api_key": "••••••••" if api_key else None,
        "base_url": base_url,
        "answer": answer,
        "citations": citations,
        "hallucination_analysis": hallucination_analysis,
        "ingested": ingested
    }
    INTERACTION_HISTORY.append(entry)
    # Keep only last 100 entries to prevent memory issues
    if len(INTERACTION_HISTORY) > 100:
        INTERACTION_HISTORY.pop(0)


@router.get("/ui")
async def ui_home(request: Request):
    # Ensure consistent spacing by sending empty strings for optional fields
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer": None,
            "citations": [],
            "q": "",
            "ingested": request.query_params.get("ingested", ""),
            "first_names": ", ".join((LAST_INGEST.get("names") or [])[:FIRST_N]) if LAST_INGEST else "",
            "has_download": bool(LAST_INGEST.get("names_text")) if LAST_INGEST else False,
            "stats": LAST_INGEST.get("stats", {}),
        },
    )


@router.get("/ui/names.txt")
async def download_names() -> str:
    return (LAST_INGEST.get("names_text") or "").strip()


@router.get("/ui/history")
async def ui_history(request: Request):
    """Display interaction history page."""
    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "history": INTERACTION_HISTORY
        },
    )


@router.post("/ui/query")
async def ui_query(request: Request):
    global LAST_INGEST  # Declare global at the top
    try:
        print("Starting UI form processing...")
        form = await request.form()
        action = form.get("action", "query")
        
        # Get form values for persistence
        model = form.get("model", "llama-3.1-8b-instant")
        api_key = form.get("api_key", "").strip()
        base_url = form.get("base_url", "").strip()
        query = form.get("q", "").strip()
        
        # Handle upload action
        if action == "upload":
            print("Processing upload action...")
            file = form.get("file")
            if not file or not hasattr(file, 'filename'):
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "ingested": LAST_INGEST.get("count", 0) if LAST_INGEST else 0,
                    "first_names": ", ".join((LAST_INGEST.get("names") or [])[:FIRST_N]) if LAST_INGEST else "",
                    "has_download": bool(LAST_INGEST.get("names_text")) if LAST_INGEST else False,
                    "stats": LAST_INGEST.get("stats", {}),
                    "q": query,
                    "model": model,
                    "api_key": api_key,
                    "base_url": base_url,
                    "error": "Please select a file to upload"
                })
            
            try:
                # Save into a unique per-upload directory to avoid mixing with sample datasets
                uploads_root = Path(__file__).resolve().parent.parent / "data/raw/uploads"
                upload_dir = uploads_root / uuid.uuid4().hex
                upload_dir.mkdir(parents=True, exist_ok=True)
                target = upload_dir / file.filename
                target.write_bytes(await file.read())
                
                # Lazy import to avoid circular import with app.api
                from importlib import import_module
                api = import_module("app.api")
                
                # If a directory is provided, ETL will run over just this upload directory
                _ = await api.ingest(api.IngestRequest(dataset_path=str(upload_dir)))
                
                # Derive count, names, and stats from the in-memory db for accuracy
                db: List[Dict[str, Any]] = list(getattr(api, "db_stub", []) or [])
                count = len(db)
                names: List[str] = []
                handles_set = set()
                hashtag_counter: Counter[str] = Counter()
                
                for r in db:
                    name = r.get("name") or r.get("metadata", {}).get("name")
                    if name:
                        names.append(str(name))
                    handle = r.get("handle") or r.get("metadata", {}).get("handle")
                    if handle:
                        handles_set.add(str(handle))
                    post = r.get("sample_post") or r.get("metadata", {}).get("sample_post") or ""
                    for token in str(post).split():
                        if token.startswith("#") and len(token) > 1:
                            tag = "#" + "".join(ch for ch in token[1:].lower() if ch.isalnum() or ch == "_")
                            if len(tag) > 1:
                                hashtag_counter[tag] += 1
                
                # Update last ingest cache
                LAST_INGEST = {
                    "count": count,
                    "dataset_name": file.filename,
                    "names": names,
                    "names_text": "\n".join(names),
                    "stats": {
                        "unique_handles": len(handles_set),
                        "top_hashtags": hashtag_counter.most_common(10),
                    },
                }
                
                print(f"Upload successful: {count} records ingested")
                
                # Log the upload interaction
                log_interaction(
                    action="upload",
                    dataset=file.filename,
                    ingested=count
                )
                
            except Exception as e:
                print(f"Upload error: {e}")
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "ingested": LAST_INGEST.get("count", 0) if LAST_INGEST else 0,
                    "first_names": ", ".join((LAST_INGEST.get("names") or [])[:FIRST_N]) if LAST_INGEST else "",
                    "has_download": bool(LAST_INGEST.get("names_text")) if LAST_INGEST else False,
                    "stats": LAST_INGEST.get("stats", {}),
                    "q": query,
                    "model": model,
                    "api_key": api_key,
                    "base_url": base_url,
                    "error": f"Upload failed: {str(e)}"
                })
        
        # Handle query action
        elif action == "query":
            if not query:
                print("Empty query, returning form...")
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "ingested": LAST_INGEST.get("count", 0) if LAST_INGEST else 0,
                    "first_names": ", ".join((LAST_INGEST.get("names") or [])[:FIRST_N]) if LAST_INGEST else "",
                    "has_download": bool(LAST_INGEST.get("names_text")) if LAST_INGEST else False,
                    "stats": LAST_INGEST.get("stats", {}),
                    "q": query,
                    "model": model,
                    "api_key": api_key,
                    "base_url": base_url
                })
            
            print(f"Processing query: {query}")
            print(f"Model: {model}, API key provided: {bool(api_key)}, Base URL provided: {bool(base_url)}")
            
            # Lazy import to avoid circular import
            from importlib import import_module
            api = import_module("app.api")
            
            print("Calling API query endpoint...")
            # Call the API
            response = await api.query(api.QueryRequest(
                query=query,
                model=model if model else None,
                api_key=api_key if api_key else None,
                base_url=base_url if base_url else None
            ))
            
            print("API call successful, extracting hallucination analysis...")
            # Extract hallucination analysis
            hallucination_analysis = response.hallucination_analysis
            
            print("Rendering template with results...")
            
            # Log the query interaction
            log_interaction(
                action="query",
                dataset=LAST_INGEST.get("dataset_name", "Unknown") if LAST_INGEST else None,
                query=query,
                model=model,
                api_key=api_key,
                base_url=base_url,
                answer=response.answer,
                citations=response.citations,
                hallucination_analysis=hallucination_analysis
            )
            
            return templates.TemplateResponse("index.html", {
                "request": request,
                "ingested": LAST_INGEST.get("count", 0) if LAST_INGEST else 0,
                "first_names": ", ".join((LAST_INGEST.get("names") or [])[:FIRST_N]) if LAST_INGEST else "",
                "has_download": bool(LAST_INGEST.get("names_text")) if LAST_INGEST else False,
                "stats": LAST_INGEST.get("stats", {}),
                "q": query,
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
                "answer": response.answer,
                "citations": response.citations,
                "hallucination_analysis": hallucination_analysis
            })
        
        # Default: just return the form with current state
        return templates.TemplateResponse("index.html", {
            "request": request,
            "ingested": LAST_INGEST.get("count", 0) if LAST_INGEST else 0,
            "first_names": ", ".join((LAST_INGEST.get("names") or [])[:FIRST_N]) if LAST_INGEST else "",
            "has_download": bool(LAST_INGEST.get("names_text")) if LAST_INGEST else False,
            "stats": LAST_INGEST.get("stats", {}),
            "q": query,
            "model": model,
            "api_key": api_key,
            "base_url": base_url
        })
        
    except Exception as e:
        print(f"Error in UI form processing: {e}")
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "ingested": LAST_INGEST.get("count", 0) if LAST_INGEST else 0,
            "first_names": ", ".join((LAST_INGEST.get("names") or [])[:FIRST_N]) if LAST_INGEST else "",
            "has_download": bool(LAST_INGEST.get("names_text")) if LAST_INGEST else False,
            "stats": LAST_INGEST.get("stats", {}),
            "q": query if 'query' in locals() else "",
            "model": model if 'model' in locals() else "llama-3.1-8b-instant",
            "api_key": api_key if 'api_key' in locals() else "",
            "base_url": base_url if 'base_url' in locals() else "",
            "error": f"Error: {str(e)}"
        })


# Dedicated About page
@router.get("/ui/about")
async def ui_about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


# How this works page
@router.get("/ui/how-it-works")
async def ui_how_it_works(request: Request):
    return templates.TemplateResponse("how-it-works.html", {"request": request})


# Updates page
@router.get("/ui/updates")
async def ui_updates(request: Request):
    return templates.TemplateResponse("updates.html", {"request": request})

