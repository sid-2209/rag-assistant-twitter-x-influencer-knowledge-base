from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .config import FAISS_MODE, VECTOR_BACKEND, CHROMA_DIR, CHROMA_COLLECTION

try:  # Optional FAISS import
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - optional import
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

try:  # Optional OpenAI import
    from openai import OpenAI  # OpenAI SDK v1
except Exception:  # pragma: no cover - optional import
    OpenAI = None  # type: ignore


EMBED_DIMENSION = 1536  # matches text-embedding-3-small output dimension


def _normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _hashed_embedding(text: str, dim: int = EMBED_DIMENSION) -> List[float]:
    # Simple, deterministic hashing-based embedding for offline/dev use
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    return _normalize(vec)


def get_embedding(text: str) -> List[float]:
    """Return embedding for text using OpenAI if available, otherwise a hashed fallback.

    Uses model: text-embedding-3-small (dimension 1536)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and OpenAI is not None:  # attempt real embeddings
        client = OpenAI()
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        emb = resp.data[0].embedding
        return _normalize(list(emb))
    # Fallback: hashed embeddings
    return _hashed_embedding(text)


class VectorStore:
    """Simple vector store with cosine similarity.

    Uses FAISS if available, otherwise falls back to a NumPy-only implementation.
    """

    def __init__(self) -> None:
        # FAISS index if available; otherwise None
        self.index = None
        self.dim: int | None = None
        self.metadata_store: List[Dict[str, Any]] = []
        # Fallback storage when FAISS is unavailable
        self._matrix = None  # lazy np.ndarray[n_docs, dim]

    def _use_faiss(self) -> bool:
        mode = FAISS_MODE
        if mode in ("false", "0"):
            return False
        if mode in ("true", "1"):
            return _FAISS_AVAILABLE
        # auto
        return _FAISS_AVAILABLE

    def _ensure_index(self, dimension: int) -> None:
        if self.index is None and self._use_faiss():
            # Cosine similarity via normalized vectors + inner product
            self.index = faiss.IndexFlatIP(dimension)  # type: ignore[attr-defined]
            self.dim = dimension
        elif self.index is None and not self._use_faiss():
            self.index = None
            self.dim = dimension

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        if not docs:
            return
        vectors: List[List[float]] = []
        for doc in docs:
            text: str = str(doc.get("text", ""))
            emb = get_embedding(text)
            vectors.append(_normalize(emb))
            # store metadata
            self.metadata_store.append(doc.get("metadata", {}))
        # Initialize index if needed
        self._ensure_index(len(vectors[0]))
        import numpy as np  # local import to keep module lightweight

        xb = np.array(vectors, dtype="float32")
        if self._use_faiss() and self.index is not None:
            # type: ignore[union-attr]
            self.index.add(xb)  # type: ignore[attr-defined]
        else:
            # Append to fallback matrix
            if self._matrix is None:
                self._matrix = xb
            else:
                self._matrix = np.vstack([self._matrix, xb])

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        import numpy as np

        use_faiss = self._use_faiss()
        if (use_faiss and self.index is None) or ((not use_faiss) and self._matrix is None):
            return []

        q_emb = _normalize(get_embedding(query))
        xq = np.array([q_emb], dtype="float32")

        results: List[Dict[str, Any]] = []
        if use_faiss and self.index is not None:
            scores, idxs = self.index.search(xq, top_k)  # type: ignore[attr-defined]
            for rank, idx in enumerate(idxs[0]):
                if idx == -1:
                    continue
                meta = self.metadata_store[idx] if idx < len(self.metadata_store) else {}
                results.append({"score": float(scores[0][rank]), **meta})
            return results

        # Fallback: brute-force cosine similarity via dot product (vectors are normalized)
        assert self._matrix is not None
        sims = np.dot(self._matrix, xq[0])  # shape [n_docs]
        top_idx = np.argsort(sims)[::-1][:top_k]
        for idx in top_idx:
            meta = self.metadata_store[int(idx)] if int(idx) < len(self.metadata_store) else {}
            results.append({"score": float(sims[int(idx)]), **meta})
        return results

    def has_data(self) -> bool:
        if _FAISS_AVAILABLE and self.index is not None:
            try:
                # type: ignore[union-attr]
                return bool(self.index.ntotal)  # type: ignore[attr-defined]
            except Exception:
                return False
        if self._matrix is not None:
            try:
                import numpy as np  # noqa: F401
                return int(getattr(self._matrix, "shape", [0])[0]) > 0
            except Exception:
                return False
        return False

    def save(self, target_dir: str | Path) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, Any] = {}
        if _FAISS_AVAILABLE and self.index is not None:
            backend = "faiss"
            try:
                faiss.write_index(self.index, str(target / "index.faiss"))  # type: ignore[attr-defined]
            except Exception:
                backend = "none"
        else:
            backend = "numpy"
            try:
                import numpy as np
                if self._matrix is not None:
                    np.save(target / "matrix.npy", self._matrix)
            except Exception:
                backend = "none"

        with (target / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, ensure_ascii=False)

        manifest.update(
            {
                "backend": backend,
                "dim": self.dim,
                "count": len(self.metadata_store),
            }
        )
        with (target / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f)

    @classmethod
    def load(cls, source_dir: str | Path) -> Optional["VectorStore"]:
        source = Path(source_dir)
        manifest_path = source / "manifest.json"
        metadata_path = source / "metadata.json"
        if not manifest_path.exists() or not metadata_path.exists():
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            return None

        backend = manifest.get("backend")
        vs = cls()
        vs.dim = manifest.get("dim")
        vs.metadata_store = list(metadata) if isinstance(metadata, list) else []

        if backend == "faiss":
            if not _FAISS_AVAILABLE:
                return None
            index_path = source / "index.faiss"
            if not index_path.exists():
                return None
            try:
                vs.index = faiss.read_index(str(index_path))  # type: ignore[attr-defined]
            except Exception:
                return None
            return vs

        if backend == "numpy":
            import numpy as np
            matrix_path = source / "matrix.npy"
            if not matrix_path.exists():
                return None
            try:
                vs._matrix = np.load(matrix_path)
                if vs._matrix is not None and getattr(vs._matrix, "shape", None):
                    vs.dim = int(vs._matrix.shape[1])
            except Exception:
                return None
            return vs

        return None


class ChromaVectorStore:
    """Optional Chroma-backed vector store wrapper."""

    def __init__(self) -> None:
        try:
            import chromadb  # type: ignore
        except Exception:
            raise RuntimeError("Chroma is not installed. Set VECTOR_BACKEND=native or install chromadb.")
        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))  # type: ignore[attr-defined]
        self._collection = self._client.get_or_create_collection(name=CHROMA_COLLECTION)  # type: ignore[attr-defined]

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        if not docs:
            return
        texts = [str(d.get("text", "")) for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]
        ids = [str(m.get("id", i)) for i, m in enumerate(metadatas)]
        self._collection.add(documents=texts, metadatas=metadatas, ids=ids)  # type: ignore[attr-defined]

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        res = self._collection.query(query_texts=[query], n_results=top_k)  # type: ignore[attr-defined]
        results: List[Dict[str, Any]] = []
        for idx in range(len(res.get("ids", [[]])[0])):
            meta = res.get("metadatas", [[{}]])[0][idx] or {}
            distance = res.get("distances", [[0]])[0][idx] if res.get("distances") else 0.0
            score = float(1.0 - distance) if isinstance(distance, (int, float)) else 0.0
            results.append({"score": score, **meta})
        return results

    def save(self, target_dir: str | Path) -> None:
        _ = (target_dir)  # Chroma persists automatically

    @classmethod
    def load(cls, source_dir: str | Path) -> Optional["ChromaVectorStore"]:
        # Persistence handled by PersistentClient; simply construct a new instance
        try:
            return cls()
        except Exception:
            return None


def create_vector_store() -> Any:
    if VECTOR_BACKEND == "chroma":
        return ChromaVectorStore()
    return VectorStore()
