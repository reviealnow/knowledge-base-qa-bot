# Knowledge Base Q&A Bot

A lightweight customer support Q&A bot built with **FastAPI + BM25** (no embeddings, no vector DB).

## Features

- **BM25 keyword retrieval** — pure Python, no ML dependencies
- **Citation grounding** — every answer cites its source (`filename#heading`)
- **Score threshold** — out-of-scope questions get a clean refusal, not hallucinations
- **SSE streaming** — tokens stream to the browser in real time
- **Conversation memory** — server-side session history lets the LLM handle follow-up questions while KB grounding stays controlled by BM25
- **Browser UI** — single-page HTML, no framework, works out of the box

## Architecture

```
docs/*.md  →  BM25 index  →  ranked sections  →  gpt-4o-mini  →  cited answer
                                                       ↑
                                              session history (last 3 turns)
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full topology and request flow.

## Quickstart

```bash
# 1. Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Start the server
export OPENAI_API_KEY="sk-..."
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. Build the index (once per docs change)
curl -X POST http://localhost:8000/index

# 4. Open the browser UI
open http://localhost:8000
```

## Project Structure

```
├── app/
│   ├── main.py        # FastAPI app, lifespan, static files
│   ├── routes.py      # GET /health, POST /index, POST /chat, POST /chat/stream
│   ├── schemas.py     # Pydantic models (ChatRequest, ChatResponse, SourceInfo)
│   ├── indexer.py     # Markdown parser, BM25 scoring, .kb/index.json persistence
│   └── retrieval.py   # Query logic, session memory, OpenAI streaming
├── static/
│   └── index.html     # Browser UI (fetch + ReadableStream SSE parser)
├── docs/              # Sample knowledge base (Markdown files)
│   ├── account_help.md
│   ├── refund_policy.md
│   └── shipping_faq.md
├── ARCHITECTURE.md    # System topology and request flow diagrams
└── requirements.txt
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/index` | Parse `docs/*.md` → rebuild BM25 index |
| `POST` | `/chat` | Sync Q&A with session memory |
| `POST` | `/chat/stream` | SSE streaming Q&A with session memory |

### Example

```bash
# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How long do refunds take?"}'

# Follow-up using the same session_id returned above
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What about the processing fee?", "session_id": "<uuid>"}'
```

## Design Decisions

**Why BM25 instead of vector embeddings?**  
Zero infrastructure overhead. No embedding model, no vector DB. For a bounded knowledge base (tens of documents), BM25 keyword matching is fast, explainable, and produces reliable citation scores.

**Why does history not affect BM25 retrieval?**  
Conversation history is only injected into the LLM message list — it never changes which sections BM25 retrieves. This keeps source grounding honest: the answer is always grounded in the current question's top-ranked sections, not past context.

## Model

Default: `gpt-4o-mini`. Override with `OPENAI_MODEL` env var.
