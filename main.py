from fastapi import FastAPI
from app import routes

app = FastAPI(title="Twitter Influencer Assistant")

# include endpoints from routes.py
app.include_router(routes.router)


