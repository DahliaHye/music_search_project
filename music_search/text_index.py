"""
Elasticsearch 대신 로컬 JSON + (선택) 단순 역인덱스.
서버가 있으면 elasticsearch 패키지로 동일 인터페이스를 구현할 수 있습니다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def build_inverted_index(meta: list[dict[str, Any]]) -> dict[str, list[int]]:
    """단어 토큰 → 트랙 인덱스."""
    inv: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        text = f"{m.get('filename', '')} {m.get('text_tags', '')} {m.get('guess_artist', '')}"
        for tok in re.split(r"\W+", text.lower()):
            if len(tok) < 2:
                continue
            inv.setdefault(tok, []).append(i)
    return inv


def save_elasticsearch_like_bundle(
    dir_path: Path,
    meta: list[dict[str, Any]],
    inv: dict[str, list[int]],
) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "tracks.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dir_path / "inverted_index.json").write_text(
        json.dumps({k: v for k, v in inv.items()}, ensure_ascii=False),
        encoding="utf-8",
    )
