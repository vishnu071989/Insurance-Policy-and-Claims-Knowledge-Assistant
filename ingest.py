"""Document extraction -> chunking -> embedding -> persisted index."""
import re, sys
from pathlib import Path
import numpy as np
from access import ROLES
from config import CORPUS_DIR, INDEX_DIR, CHUNK_CHARS, CHUNK_OVERLAP
from vectorstore import VectorStore

def parse_front_matter(raw: str):
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)$", raw, re.S)
    if not m:
        raise ValueError("missing front matter")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    for k in ("doc_id", "title", "allowed_roles"):
        if k not in meta:
            raise ValueError(f"front matter missing '{k}'")
    roles = [r.strip() for r in meta["allowed_roles"].strip("[]").split(",") if r.strip()]
    if not roles or set(roles) - ROLES:
        raise ValueError(f"invalid allowed_roles: {roles}")
    meta["allowed_roles"] = roles
    return meta, m.group(2)

def extract(path: Path) -> str:
    if path.suffix.lower() == ".pdf":  # needs sidecar <file>.pdf.meta with the front-matter keys
        from pypdf import PdfReader
        meta = Path(str(path) + ".meta").read_text(encoding="utf-8")
        text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
        return f"---\n{meta.strip()}\n---\n{text}"
    return path.read_text(encoding="utf-8")

def chunk_text(body: str, max_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    sections, title, buf = [], "Introduction", []
    for line in body.splitlines():
        if re.match(r"^#{1,4}\s", line):
            if buf:
                sections.append((title, "\n".join(buf).strip()))
            buf, title = [], line.lstrip("# ").strip()
        else:
            buf.append(line)
    if buf:
        sections.append((title, "\n".join(buf).strip()))
    out = []
    for title, text in sections:
        if not text:
            continue
        start = 0
        while True:
            out.append((title, text[start:start + max_chars]))
            if start + max_chars >= len(text):
                break
            start += max_chars - overlap
    return out

def build_chunks(corpus_dir: Path):
    chunks = []
    files = sorted(p for p in Path(corpus_dir).iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"})
    for f in files:
        meta, body = parse_front_matter(extract(f))
        for i, (section, text) in enumerate(chunk_text(body)):
            chunks.append({"chunk_id": f"{meta['doc_id']}#{i}", "doc_id": meta["doc_id"],
                           "title": meta["title"], "section": section,
                           "allowed_roles": meta["allowed_roles"], "text": text})
    return chunks

def run(corpus_dir=CORPUS_DIR, index_dir=INDEX_DIR, embed_fn=None):
    if embed_fn is None:
        from bedrock_client import embed as embed_fn
    chunks = build_chunks(corpus_dir)
    vecs = []
    for c in chunks:
        v = np.asarray(embed_fn(f"{c['title']} - {c['section']}\n{c['text']}"), dtype=np.float32)
        vecs.append(v / (np.linalg.norm(v) or 1.0))
    store = VectorStore(chunks, np.vstack(vecs))
    store.save(index_dir)
    return store

if __name__ == "__main__":
    s = run()
    print(f"Indexed {len(s.chunks)} chunks from {len({c['doc_id'] for c in s.chunks})} documents -> {INDEX_DIR}")
