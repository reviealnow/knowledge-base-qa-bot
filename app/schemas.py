from pydantic import BaseModel


class IndexResponse(BaseModel):
    files_indexed: int
    sections_indexed: int


class ChatRequest(BaseModel):
    query: str
    score_threshold: float = 0.5
    session_id: str | None = None


class SourceInfo(BaseModel):
    source: str
    heading: str
    score: float
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    threshold_applied: bool = False
    session_id: str
