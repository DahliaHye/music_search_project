"""
하이브리드: (1) 트랙 태그 문서에 대한 TF-IDF / 문장 임베딩 유사도
         (2) 선택: 시드 트랙 오디오 임베딩과의 코사인 (콘텐츠 기반)
         + 아티스트·무드 부스트.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .audio_features import l2_normalize
from .nlp_query import MOOD_SYNONYMS, ParsedQuery


def load_meta(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _st_query_vs_docs(query_text: str, docs: list[str]) -> np.ndarray | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        emb_q = model.encode([query_text], normalize_embeddings=True)
        emb_d = model.encode(docs, normalize_embeddings=True)
        return cosine_similarity(emb_q, emb_d).ravel()
    except Exception:
        return None


def hybrid_scores(
    parsed: ParsedQuery,
    *,
    embeddings: np.ndarray,
    descriptions: list[str],
    track_meta: list[dict[str, Any]],
    seed_track_idx: int | None = None,
    audio_embeddings: np.ndarray | None = None,
    alpha_tfidf: float = 0.4,
    alpha_st: float = 0.35,
    alpha_audio: float = 0.25,
    artist_boost: float = 0.22,
    mood_boost: float = 0.07,
) -> tuple[list[tuple[int, float]], dict[str, Any]]:
    n = len(descriptions)
    if n == 0:
        return [], {}

    vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
    doc_m = vec.fit_transform(descriptions)
    qv = vec.transform([parsed.text_for_tfidf])
    tfidf_sim = cosine_similarity(qv, doc_m).ravel()

    st_sim = _st_query_vs_docs(parsed.text_for_tfidf, descriptions)
    use_st = st_sim is not None
    if not use_st:
        st_sim = tfidf_sim.copy()

    audio_src = audio_embeddings if audio_embeddings is not None else embeddings
    emb_n = np.stack([l2_normalize(audio_src[i]) for i in range(n)])
    if seed_track_idx is not None and 0 <= seed_track_idx < n:
        seed = emb_n[seed_track_idx]
        audio_sim = (emb_n @ seed).astype(np.float64)
    else:
        audio_sim = np.ones(n, dtype=np.float64) / n

    # 시드 없으면 텍스트 쪽에 가중 (오디오 항은 균등이라 의미 없음 → 비중 0에 가깝게)
    if seed_track_idx is None:
        s = alpha_tfidf + alpha_st
        at = alpha_tfidf / s
        ast = alpha_st / s
        combined = at * tfidf_sim + ast * st_sim
        used_audio = False
    else:
        combined = (
            alpha_tfidf * tfidf_sim
            + alpha_st * st_sim
            + alpha_audio * audio_sim
        )
        used_audio = True

    if parsed.artist_hint:
        hint = parsed.artist_hint.lower()
        for i, m in enumerate(track_meta):
            blob = f"{m.get('filename', '')} {m.get('guess_artist', '')}".lower()
            if hint in blob:
                combined[i] += artist_boost

    for mood in parsed.mood_boosts:
        keys = MOOD_SYNONYMS.get(mood, [])
        for i, m in enumerate(track_meta):
            tags = (m.get("text_tags") or "").lower()
            if any(k.lower() in tags for k in keys):
                combined[i] += mood_boost

    order = np.argsort(-combined)
    ranked = [(int(i), float(combined[i])) for i in order]
    debug = {
        "alpha_tfidf": alpha_tfidf,
        "alpha_st": alpha_st,
        "alpha_audio": alpha_audio,
        "used_sentence_transformers": use_st,
        "used_seed_audio": seed_track_idx is not None,
        "used_vocal_embedding_for_audio": audio_embeddings is not None,
    }
    return ranked, debug


def content_similarity_rank(
    embeddings: np.ndarray,
    seed_idx: int,
) -> list[tuple[int, float]]:
    """순수 콘텐츠 벡터 코사인 유사도 (시드 트랙 기준)."""
    n = embeddings.shape[0]
    emb_n = np.stack([l2_normalize(embeddings[i]) for i in range(n)])
    seed = emb_n[seed_idx]
    sim = (emb_n @ seed).astype(np.float64)
    order = np.argsort(-sim)
    return [(int(i), float(sim[i])) for i in order]
