from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .indexer import load_index
from . import indexer_vector
from .routes import router

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_index()
    try:
        indexer_vector.load_index()
    except Exception as e:
        print(f"[vector] Could not load index on startup: {e}", flush=True)
    yield


app = FastAPI(title="Knowledge Base Q&A Bot", lifespan=lifespan)
app.include_router(router)


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
