"""audio 파일명에서 가수·곡 후보 추출 (acrcloud_recognize 와 동일 규칙)."""
from __future__ import annotations

from pathlib import Path


def stem_without_audio_suffix(filename: str) -> str:
    stem = Path(filename).stem
    if stem.endswith("_audio"):
        return stem[: -len("_audio")]
    return stem


def guess_artist_title(filename: str) -> tuple[str, str]:
    base = stem_without_audio_suffix(filename)
    if "_" not in base:
        return ("", base.strip())
    artist, title = base.split("_", 1)
    return (artist.strip(), title.strip())
