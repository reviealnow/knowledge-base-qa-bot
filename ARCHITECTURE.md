# Knowledge Base Q&A Bot — Architecture

## System Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (Client)                         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  static/index.html                                        │  │
│  │                                                           │  │
│  │  sessionId (module var)  ←── event: session_id (SSE)     │  │
│  │  turnCount               ←── event: done (SSE)           │  │
│  │                                                           │  │
│  │  POST /chat/stream  ──► { query, session_id }            │  │
│  │                                                           │  │
│  │  SSE event stream (ReadableStream):                       │  │
│  │    session_id → sources → threshold_applied              │  │
│  │    → token × N → done                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP / SSE
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Server (uvicorn)                      │
│                                                                  │
│  ┌──────────────┐    ┌───────────────────────────────────────┐  │
│  │  routes.py   │    │  retrieval.py                         │  │
│  │              │    │                                        │  │
│  │  POST /index ├───►│  build_index()                        │  │
│  │              │    │    docs/*.md → .kb/index.json         │  │
│  │  POST /chat  │    │                                        │  │
│  │  POST /chat/ ├───►│  _get_session(session_id)             │  │
│  │    stream    │    │    ┌──────────────────────────────┐   │  │
│  │              │    │    │  SESSIONS dict               │   │  │
│  │  GET /health │    │    │  { uuid → deque(maxlen=6) }  │   │  │
│  └──────────────┘    │    └──────────┬───────────────────┘   │  │
│                      │               │ history (last 3 turns) │  │
│  ┌──────────────┐    │               ▼                        │  │
│  │  schemas.py  │    │  indexer.search(query, k=3)            │  │
│  │              │    │    BM25 scoring on current question     │  │
│  │  ChatRequest │    │    → ranked sections (score ≥ 0.5)     │  │
│  │  ChatResponse│    │               │                        │  │
│  │  SourceInfo  │    │               ▼                        │  │
│  └──────────────┘    │  _build_prompt(question, ranked)       │  │
│                      │    CONTEXT block + QUESTION            │  │
│                      │               │                        │  │
│                      │               ▼                        │  │
│                      │  OpenAI API (gpt-4o-mini)              │  │
│                      │    messages:                           │  │
│                      │      [system] SYSTEM_PROMPT            │  │
│                      │      [user]   turn-1 raw question      │  │
│                      │      [asst]   turn-1 answer            │  │
│                      │      ...history...                     │  │
│                      │      [user]   CONTEXT + question       │  │
│                      │    stream=True → SSE tokens            │  │
│                      │               │                        │  │
│                      │               ▼                        │  │
│                      │  _save_turn(sid, question, answer)     │  │
│                      └───────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         File System                              │
│                                                                  │
│  knowledge_base_qa_bot/                                          │
│  ├── docs/                   ← source of truth (Markdown KB)    │
│  │   ├── account_help.md                                         │
│  │   ├── refund_policy.md                                        │
│  │   └── shipping_faq.md                                         │
│  ├── my_kb/                  ← FastAPI app root                  │
│  │   ├── app/                                                     │
│  │   │   ├── main.py         ← FastAPI app, lifespan             │
│  │   │   ├── routes.py       ← endpoints                         │
│  │   │   ├── schemas.py      ← Pydantic models                   │
│  │   │   ├── indexer.py      ← BM25 engine + Markdown parser     │
│  │   │   └── retrieval.py    ← query logic + SESSIONS store      │
│  │   ├── static/             │
│  │   │   └── index.html      ← single-page browser UI            │
│  │   └── .kb/               │
│  │       └── index.json      ← pre-built BM25 index (cache)      │
│  └── ARCHITECTURE.md         ← this file                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Request Flow — POST /chat/stream

```
Client                    FastAPI                  OpenAI
  │                          │                        │
  │── POST /chat/stream ────►│                        │
  │   { query, session_id }  │                        │
  │                          │── _get_session() ──┐   │
  │                          │   (create/lookup)  │   │
  │                          │◄───────────────────┘   │
  │                          │                        │
  │◄── event: session_id ────│                        │
  │                          │                        │
  │                          │── indexer.search() ─┐  │
  │                          │   BM25 on query     │  │
  │                          │◄────────────────────┘  │
  │                          │                        │
  │◄── event: sources ───────│  (ranked sections)     │
  │◄── event: threshold ─────│                        │
  │                          │                        │
  │                          │── chat.completions ────►│
  │                          │   [system+history+ctx] │
  │                          │                        │
  │◄── event: token ─────────│◄── stream chunk ───────│
  │◄── event: token ─────────│◄── stream chunk ───────│
  │        ...               │        ...             │
  │◄── event: done ──────────│── _save_turn() ─────┐  │
  │                          │   (update SESSIONS) │  │
  │                          │◄────────────────────┘  │
```

---

## Session Memory Model

```
SESSIONS (in-memory dict)
┌──────────────────────────────────────────────────────┐
│  "a1f9ac05-..."  →  deque(maxlen=6)                  │
│                     ┌──────────────────────────────┐ │
│                     │ [user]  "How long do refunds" │ │
│                     │ [asst]  "5-7 business days…"  │ │
│                     │ [user]  "What about the fee?" │ │
│                     │ [asst]  "There is a 3% fee…"  │ │
│                     │ [user]  "Is it waived?"        │ │
│                     │ [asst]  "Yes, for orders >…"  │ │
│                     └──────────────────────────────┘ │
│                       maxlen=6 → 3 turns rolling     │
│                                                       │
│  "866a0642-..."  →  deque(maxlen=6)   ← new session  │
│                     [ empty ]                         │
└──────────────────────────────────────────────────────┘

Key invariant: BM25 search always uses the CURRENT question only.
History influences LLM context — it never changes which sources are retrieved.
```

---

## Endpoint Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/index` | Parse `docs/*.md` → rebuild `.kb/index.json` |
| `POST` | `/chat` | Sync Q&A with session memory |
| `POST` | `/chat/stream` | SSE streaming Q&A with session memory |
| `GET`  | `/` | Redirect to `/static/index.html` |
| `GET`  | `/static/index.html` | Browser UI |
