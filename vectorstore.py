import json
from pathlib import Path
import numpy as np

class VectorStore:
    """Persisted cosine-similarity index. Role filtering happens BEFORE ranking."""
    def __init__(self, chunks, vectors):
        self.chunks, self.vectors = chunks, np.asarray(vectors, dtype=np.float32)

    @classmethod
    def load(cls, index_dir: Path):
        cj, vn = Path(index_dir) / "chunks.json", Path(index_dir) / "vectors.npy"
        if not cj.exists() or not vn.exists():
            raise FileNotFoundError("Index not found. Run: python ingest.py")
        return cls(json.loads(cj.read_text(encoding="utf-8")), np.load(vn))

    def save(self, index_dir: Path):
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        (Path(index_dir) / "chunks.json").write_text(json.dumps(self.chunks, indent=1), encoding="utf-8")
        np.save(Path(index_dir) / "vectors.npy", self.vectors)

    def search(self, qvec, role: str, k: int):
        q = np.asarray(qvec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        idx = [i for i, c in enumerate(self.chunks) if role in c["allowed_roles"]]
        if not idx:
            return []
        sims = self.vectors[idx] @ q
        order = np.argsort(-sims)[:k]
        return [{**self.chunks[idx[j]], "score": float(sims[j])} for j in order]
