"""
딥러닝 대체·보강용: 학습된 음악 CNN/Transformer 대신,
여기서는 (1) sklearn PCA 로 차원 축소한 '압축 임베딩' 또는
(2) 향후 torch 모델을 끼울 수 있는 동일 shape 인터페이스.

실서비스에서는 MusicNN, CLAP, EfficientAT 등으로 교체하면 됩니다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def pca_projection(
    X: np.ndarray,
    n_components: int = 32,
    state_path: Path | None = None,
) -> tuple[np.ndarray, Any]:
    """
    X: (n, d) 스케일된 특징.
    반환: (n, n_components) 임베딩, 학습된 PCA 객체.
    """
    from sklearn.decomposition import PCA

    n_comp = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    z = pca.fit_transform(X)
    if state_path is not None:
        import pickle

        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_bytes(pickle.dumps(pca))
    return z.astype(np.float64), pca


def load_pca(path: Path) -> Any | None:
    if not path.is_file():
        return None
    import pickle

    return pickle.loads(path.read_bytes())
