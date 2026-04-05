"""
FAISS(IndexFlatIP) 또는 numpy 브루트포스로 L2 정규화 코사인 유사도 검색.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

FAISS_AVAILABLE = False
try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore


def build_index_ip(embeddings_normalized: np.ndarray) -> Any:
    """embeddings_normalized: (n, d) L2 단위 벡터."""
    x = np.ascontiguousarray(embeddings_normalized.astype("float32"))
    d = x.shape[1]
    if FAISS_AVAILABLE:
        index = faiss.IndexFlatIP(d)  # type: ignore
        index.add(x)
        return index
    return x


def search_ip(
    index: Any,
    query_normalized: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """scores: inner product (= cosine for unit vectors), indices."""
    q = np.ascontiguousarray(query_normalized.reshape(1, -1).astype("float32"))
    if FAISS_AVAILABLE:
        scores, idx = index.search(q, k)  # type: ignore
        return scores[0], idx[0]
    mat = index @ q.T
    sim = mat.ravel()
    part = np.argpartition(-sim, min(k, len(sim) - 1))[:k]
    order = part[np.argsort(-sim[part])]
    return sim[order], order


def save_faiss_index(index: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if FAISS_AVAILABLE:
        faiss.write_index(index, str(path))  # type: ignore


def load_faiss_index(path: Path, d: int) -> Any:
    if FAISS_AVAILABLE and path.is_file():
        return faiss.read_index(str(path))  # type: ignore
    return None
