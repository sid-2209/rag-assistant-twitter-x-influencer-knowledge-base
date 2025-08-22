from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app import routes
from app import webui

app = FastAPI(title="Twitter Influencer Assistant")

# Mount static files
static_dir = Path(__file__).resolve().parent / "app" / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# include endpoints from routes.py
app.include_router(routes.router)

# include web UI endpoints
app.include_router(webui.router)


