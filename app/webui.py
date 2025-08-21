from __future__ import annotations

from pathlib import Path
import uuid
from urllib.parse import quote_plus
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates


templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

router = APIRouter()

# In-memory cache of the last ingest for UI display/download
LAST_INGEST: Dict[str, Any] = {}
FIRST_N: int = 20


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


@router.post("/ui/upload")
async def ui_upload(request: Request, file: UploadFile = File(...)):
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
        from collections import Counter
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
        global LAST_INGEST
        LAST_INGEST = {
            "names": names,
            "names_text": "\n".join(names),
            "stats": {
                "unique_handles": len(handles_set),
                "top_hashtags": hashtag_counter.most_common(10),
            },
        }
        return RedirectResponse(
            url=f"/ui?ingested={count}",
            status_code=303,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@router.get("/ui/names.txt")
async def download_names() -> str:
    return (LAST_INGEST.get("names_text") or "").strip()


@router.post("/ui/query")
async def ui_query(request: Request, q: str = Form(""), model: str = Form("gpt-4o-mini"), api_key: str = Form(""), base_url: str = Form("") ):
    # Lazy import to avoid circular import with app.api
    from importlib import import_module
    api = import_module("app.api")
    # Temporarily override env for request-scoped API key if provided
    import os
    prev_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    qr = api.QueryRequest(query=q, implementation="vanilla", model=model, api_key=api_key or None, base_url=base_url or None)
    result = await api.query(qr)
    # Restore
    if api_key:
        if prev_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = prev_key
    context: Dict[str, Any] = {
        "request": request,
        "answer": result.answer,
        "citations": result.citations,
        "q": q,
        "model": model,
        "base_url": base_url or "",
    }
    return templates.TemplateResponse("index.html", context)


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

