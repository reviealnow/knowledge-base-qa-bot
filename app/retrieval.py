import json
import os
import uuid
from collections import deque
from collections.abc import Generator

from openai import OpenAI

from . import indexer

SYSTEM_PROMPT = """You are a customer support assistant with access to a knowledge base.

Rules (strictly enforced):
1. Answer ONLY using the CONTEXT provided below.
2. Every factual claim must cite its source using the format: [filename#heading].
3. If the CONTEXT does not contain the answer, respond exactly: "I cannot confirm from the knowledge base."
4. Never guess, infer, or use outside knowledge. No exceptions."""

_client: OpenAI | None = None
SESSIONS: dict[str, deque] = {}
_MAX_TURNS = 3  # keep last 3 Q&A pairs (6 messages)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _get_session(session_id: str | None) -> tuple[str, list]:
    """Return (session_id, history_messages). Creates a new session if needed."""
    if not session_id or session_id not in SESSIONS:
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = deque(maxlen=_MAX_TURNS * 2)
    return session_id, list(SESSIONS[session_id])


def _save_turn(session_id: str, question: str, answer: str) -> None:
    SESSIONS[session_id].append({"role": "user", "content": question})
    SESSIONS[session_id].append({"role": "assistant", "content": answer})


def _build_prompt(query: str, ranked: list[tuple]) -> str:
    blocks = []
    for section, score in ranked:
        heading_md = "\n".join(
            f"{'#' * (i + 1)} {h}" for i, h in enumerate(section.heading_path)
        )
        blocks.append(
            f"[Source: {section.id}] [BM25: {score:.2f}]\n"
            f"{heading_md}\n\n"
            f"{section.content}"
        )
    context = "\n\n---\n\n".join(blocks)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def query(question: str, score_threshold: float = 0.5, session_id: str | None = None) -> dict:
    sid, history = _get_session(session_id)

    if not indexer.sections:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
            "threshold_applied": False,
            "session_id": sid,
        }

    ranked = indexer.search(question, k=3)

    if not ranked:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
            "threshold_applied": True,
            "session_id": sid,
        }

    top_score = ranked[0][1]
    if top_score < score_threshold:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
            "threshold_applied": True,
            "session_id": sid,
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": _build_prompt(question, ranked)},
    ]

    response = _get_client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        timeout=20,
    )

    answer = response.choices[0].message.content
    _save_turn(sid, question, answer)

    sources = [
        {
            "source": s.id,
            "heading": " > ".join(s.heading_path),
            "score": round(score, 3),
            "content": s.content[:240],
        }
        for s, score in ranked
    ]

    return {
        "answer": answer,
        "sources": sources,
        "threshold_applied": False,
        "session_id": sid,
    }


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def query_stream(question: str, score_threshold: float = 0.5, session_id: str | None = None) -> Generator[str, None, None]:
    sid, history = _get_session(session_id)
    yield _sse("session_id", sid)

    if not indexer.sections:
        yield _sse("error", "The knowledge base has not been indexed yet. Call POST /index first.")
        yield _sse("done", "")
        return

    ranked = indexer.search(question, k=3)

    if not ranked:
        yield _sse("sources", json.dumps([]))
        yield _sse("threshold_applied", "true")
        yield _sse("token", "I cannot confirm from the knowledge base.")
        yield _sse("done", "")
        return

    top_score = ranked[0][1]
    if top_score < score_threshold:
        yield _sse("sources", json.dumps([]))
        yield _sse("threshold_applied", "true")
        yield _sse("token", "I cannot confirm from the knowledge base.")
        yield _sse("done", "")
        return

    sources = [
        {
            "source": s.id,
            "heading": " > ".join(s.heading_path),
            "score": round(score, 3),
            "content": s.content[:240],
        }
        for s, score in ranked
    ]
    yield _sse("sources", json.dumps(sources))
    yield _sse("threshold_applied", "false")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": _build_prompt(question, ranked)},
    ]

    stream = _get_client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        stream=True,
        timeout=30,
    )

    collected = []
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            collected.append(token)
            yield _sse("token", token)

    _save_turn(sid, question, "".join(collected))
    yield _sse("done", "")
