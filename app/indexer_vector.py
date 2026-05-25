import json
import os
import re
import shutil
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
INDEX_DIR = Path(__file__).resolve().parents[2] / ".kb" / "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

vectorstore = None
files_indexed: int = 0
chunks_indexed: int = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _get_embeddings():
    from langchain_openai import OpenAIEmbeddings
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, request_timeout=20, max_retries=1)


def _load_markdown_sections(path: Path) -> list:
    from langchain.schema import Document
    docs: list[Document] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading = path.stem.replace("_", " ").title()
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if not content:
            current_lines = []
            return
        heading_path = [h for _, h in heading_stack] or [current_heading]
        section_id = f"{path.name}#{slugify(current_heading)}"
        docs.append(Document(
            page_content="\n".join([*heading_path, content]),
            metadata={
                "source": section_id,
                "file": path.name,
                "heading": " > ".join(heading_path),
            },
        ))
        current_lines = []

    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            current_heading = match.group(2).strip()
            heading_stack = [(lvl, h) for lvl, h in heading_stack if lvl < level]
            heading_stack.append((level, current_heading))
        else:
            current_lines.append(line)
    flush()
    return docs


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, chunks_indexed
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    markdown_files = sorted(docs_dir.glob("*.md"))
    section_docs = []
    for path in markdown_files:
        section_docs.extend(_load_markdown_sections(path))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(section_docs)
    if chunks:
        print(f"[vector] Embedding {len(chunks)} chunks…", flush=True)
        vectorstore = FAISS.from_documents(chunks, _get_embeddings())
    else:
        vectorstore = None

    files_indexed = len(markdown_files)
    chunks_indexed = len(chunks)
    _save_index()
    return files_indexed, chunks_indexed


def _save_index(index_dir: Path = INDEX_DIR) -> None:
    if vectorstore is None:
        if index_dir.exists():
            shutil.rmtree(index_dir)
        return
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "files_indexed": files_indexed,
        "chunks_indexed": chunks_indexed,
    }
    (index_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[vector] FAISS index saved to {index_dir}", flush=True)


def load_index(index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, chunks_indexed
    from langchain_community.vectorstores import FAISS

    if not (index_dir / "index.faiss").exists():
        return 0, 0

    metadata: dict = {}
    meta_path = index_dir / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    stored = metadata.get("embedding_model")
    if stored and stored != EMBEDDING_MODEL:
        raise RuntimeError(f"Index uses {stored}, server uses {EMBEDDING_MODEL}")

    vectorstore = FAISS.load_local(
        str(index_dir), _get_embeddings(), allow_dangerous_deserialization=True
    )
    files_indexed = int(metadata.get("files_indexed", 0))
    chunks_indexed = int(metadata.get("chunks_indexed", 0))
    print(f"[vector] Loaded FAISS index ({chunks_indexed} chunks)", flush=True)
    return files_indexed, chunks_indexed


def search(query: str, k: int = 3) -> list:
    if vectorstore is None:
        return []
    return vectorstore.similarity_search_with_score(query, k=k)
