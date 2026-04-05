"""
대규모 외부 데이터셋·API 하이브리드용 스텁.

- Million Song Dataset: 로컬 HDF5/SQLite 경로를 환경 변수 MSD_ROOT 등으로 두고
  여기서 메타데이터만 조인하는 식으로 확장할 수 있습니다.
- Spotify: 클라이언트 자격 증명이 있을 때만 audio_features / preview URL 메타를 가져오도록 확장.
"""
from __future__ import annotations

import os
from typing import Any


def spotify_track_features_stub(track_id: str) -> dict[str, Any] | None:
    """
    환경 변수 SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET 이 있으면
    spotipy 등으로 교체해 실제 오디오 특징을 반환하도록 합니다.
    """
    if not os.environ.get("SPOTIFY_CLIENT_ID"):
        return None
    return None


def msd_join_stub(local_id: str) -> dict[str, Any] | None:
    """Million Song Dataset 매핑 테이블이 있을 때 확장."""
    return None
