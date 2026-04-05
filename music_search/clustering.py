"""특징 벡터에 KMeans로 무드 그룹(클러스터) 라벨 부여."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans


def fit_clusters(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    embeddings: (n, d) 원본(스케일링된) 벡터. 라벨과 센트로이드 반환.
    """
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels = km.fit_predict(embeddings)
    return labels.astype(np.int32), km.cluster_centers_, {
        "n_clusters": n_clusters,
        "inertia": float(km.inertia_),
    }


def save_cluster_report(
    path: Path,
    *,
    labels: np.ndarray,
    track_names: list[str],
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_c: dict[int, list[str]] = {}
    for i, lab in enumerate(labels.tolist()):
        by_c.setdefault(int(lab), []).append(track_names[i])
    out = {"meta": meta, "clusters": {str(k): v for k, v in sorted(by_c.items())}}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
