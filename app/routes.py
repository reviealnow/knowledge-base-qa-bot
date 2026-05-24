from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .indexer import build_index
from .retrieval import query, query_stream
from .schemas import ChatRequest, ChatResponse, IndexResponse

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/index", response_model=IndexResponse)
def index_docs():
    files_count, sections_count = build_index()
    return IndexResponse(files_indexed=files_count, sections_indexed=sections_count)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return query(req.query, score_threshold=req.score_threshold, session_id=req.session_id)


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    return StreamingResponse(
        query_stream(req.query, score_threshold=req.score_threshold, session_id=req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
