import os

from openai import OpenAI

from . import indexer_vector as indexer

SYSTEM_PROMPT = """You are a customer support assistant with access to a knowledge base.

Rules (strictly enforced):
1. Answer ONLY using the CONTEXT provided below.
2. Every factual claim must cite its source using the format: [filename#heading].
3. If the CONTEXT does not contain the answer, respond exactly: "I cannot confirm from the knowledge base."
4. Never guess, infer, or use outside knowledge. No exceptions."""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _build_prompt(query: str, ranked: list) -> str:
    blocks = []
    for doc, score in ranked:
        blocks.append(
            f"[Source: {doc.metadata.get('source', 'unknown')}] [L2 distance: {score:.4f}]\n"
            f"{doc.page_content}"
        )
    context = "\n\n---\n\n".join(blocks)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


# L2 distance: lower = more similar. Reject if top result is too far (> threshold).
_DEFAULT_THRESHOLD = 0.8


def query(question: str, score_threshold: float = _DEFAULT_THRESHOLD) -> dict:
    if indexer.vectorstore is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
            "threshold_applied": False,
            "strategy": "vector",
        }

    ranked = indexer.search(question, k=3)
    if not ranked:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
            "threshold_applied": True,
            "strategy": "vector",
        }

    top_distance = ranked[0][1]
    if top_distance > score_threshold:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
            "threshold_applied": True,
            "strategy": "vector",
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(question, ranked)},
    ]
    response = _get_client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        timeout=20,
    )

    answer = response.choices[0].message.content
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 4),
            "content": doc.page_content[:240],
        }
        for doc, score in ranked
    ]
    return {
        "answer": answer,
        "sources": sources,
        "threshold_applied": False,
        "strategy": "vector",
    }
