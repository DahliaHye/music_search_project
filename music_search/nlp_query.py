"""
자연어 질의를 키워드·아티스트 필터·검색용 텍스트로 분해합니다.
선택적으로 sentence-transformers가 있으면 질의 임베딩을 생성합니다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# 분위기 연관 키워드 (태그 textify 출력과 맞물리도록 영·한 혼합)
MOOD_SYNONYMS: dict[str, list[str]] = {
    "calm": ["잔잔", "차분", "평온", "calm", "relax", "잔잔한"],
    "upbeat": ["신나", "활기", "빠른", "upbeat", "energetic", "댄스"],
    "dark": ["어두", "무드", "dark", "무거운", "저음"],
    "bright": ["밝", "화사", "bright", "경쾌"],
    "melodic": ["선율", "멜로디", "melodic", "서정"],
    "rhythmic": ["리듬", "비트", "rhythmic", "그루브"],
}


@dataclass
class ParsedQuery:
    raw: str
    artist_hint: str | None
    text_for_tfidf: str
    mood_boosts: list[str]


def parse_natural_language(query: str) -> ParsedQuery:
    q = query.strip()
    artist_hint: str | None = None

    # 가수 XXX / artist: XXX / "아티스트" 패턴
    m = re.search(
        r"(?:가수|artist|아티스트)\s*[:：]?\s*['\"]?([^'\"]+?)['\"]?(?:\s|$|으로|로|를|을)",
        q,
        re.IGNORECASE,
    )
    if m:
        artist_hint = m.group(1).strip()
    else:
        m2 = re.search(r"['\"]([^'\"]{2,40})['\"]\s*(?:의\s*)?(?:곡|노래|음악)", q)
        if m2:
            artist_hint = m2.group(1).strip()

    mood_boosts: list[str] = []
    low = q.lower()
    for mood, words in MOOD_SYNONYMS.items():
        if any(w in q or w.lower() in low for w in words):
            mood_boosts.append(mood)

    # TF-IDF용: 질의 + 동의어 확장
    extra = " ".join(
        w for m in mood_boosts for w in MOOD_SYNONYMS.get(m, [])
    )
    text_for_tfidf = f"{q} {extra}"

    return ParsedQuery(
        raw=q,
        artist_hint=artist_hint,
        text_for_tfidf=text_for_tfidf,
        mood_boosts=mood_boosts,
    )


def try_embed_query(text: str) -> Any | None:
    """sentence-transformers 사용 가능 시 1차원 벡터 반환, 아니면 None."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return model.encode([text], normalize_embeddings=True)[0]
    except Exception:
        return None
