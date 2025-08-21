from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates


templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

router = APIRouter()


@router.get("/ui")
async def ui_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": None, "citations": []})


@router.post("/ui/upload")
async def ui_upload(request: Request, file: UploadFile = File(...)):
    try:
        raw_dir = Path(__file__).resolve().parent.parent / "data/raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        target = raw_dir / file.filename
        target.write_bytes(await file.read())
        # Lazy import to avoid circular import with app.api
        from importlib import import_module
        api = import_module("app.api")
        await api.ingest(api.IngestRequest(dataset_path=str(raw_dir)))
        return RedirectResponse(url="/ui", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@router.post("/ui/query")
async def ui_query(request: Request, q: str = Form("")):
    # Lazy import to avoid circular import with app.api
    from importlib import import_module
    api = import_module("app.api")
    qr = api.QueryRequest(query=q)
    result = await api.query(qr)
    context: Dict[str, Any] = {
        "request": request,
        "answer": result.answer,
        "citations": result.citations,
        "q": q,
    }
    return templates.TemplateResponse("index.html", context)


