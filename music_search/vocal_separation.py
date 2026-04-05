"""
보컬 스템 추정: 기본은 HPSS 하모닉 + 보컬 대역 대역통과(경량).
선택: 환경 변수 MUSIC_SEARCH_USE_DEMUCS=1 이고 demucs 설치 시 Demucs(htdemucs) 보컬 스템.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

# librosa는 호출부에서 import


def _bandpass_sos(y: np.ndarray, sr: int, fmin: float, fmax: float):
    from scipy import signal

    nyq = sr / 2.0
    low = max(fmin / nyq, 1e-5)
    high = min(fmax / nyq, 0.99)
    if low >= high:
        return y
    sos = signal.butter(4, [low, high], btype="band", output="sos")
    return signal.sosfiltfilt(sos, y)


def separate_vocals_hpss_bandpass(
    y: np.ndarray, sr: int, fmin: float = 180.0, fmax: float = 7800.0
) -> tuple[np.ndarray, dict[str, Any]]:
    import librosa

    y_h, y_p = librosa.effects.hpss(y)
    y_bp = _bandpass_sos(y_h, sr, fmin, fmax)
    peak_in = float(np.max(np.abs(y)) + 1e-9)
    peak_v = float(np.max(np.abs(y_bp)) + 1e-9)
    y_v = y_bp * (peak_in / peak_v)
    info: dict[str, Any] = {
        "method": "hpss_harmonic_bandpass",
        "fmin_hz": fmin,
        "fmax_hz": fmax,
        "harm_rms": float(np.sqrt(np.mean(y_h**2))),
        "perc_rms": float(np.sqrt(np.mean(y_p**2))),
        "vocal_rms": float(np.sqrt(np.mean(y_v**2))),
    }
    return y_v.astype(np.float32), info


def _separate_vocals_demucs_file(audio_path: Path, cache_dir: Path) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Demucs로 vocals.wav 생성 후 로드. 실패 시 (None, reason)."""
    demucs_exe = os.environ.get("DEMUCS_PYTHON", "").strip() or None
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_sub = cache_dir / "demucs_out"
    out_sub.mkdir(parents=True, exist_ok=True)
    cmd = [
        demucs_exe or sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        "htdemucs",
        "--two-stems",
        "vocals",
        "-o",
        str(out_sub),
        str(audio_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        return None, {"method": "demucs", "error": str(e)}

    # htdemucs / stem / vocals.wav 구조 탐색
    wavs = list(out_sub.rglob("vocals.wav"))
    if not wavs:
        return None, {"method": "demucs", "error": "vocals.wav not found"}
    import soundfile as sf

    v, sr = sf.read(wavs[0], always_2d=False)
    if v.ndim > 1:
        v = np.mean(v, axis=1)
    if sr != 22050:
        import librosa

        v = librosa.resample(v.astype(np.float32), orig_sr=sr, target_sr=22050)
        sr = 22050
    return v.astype(np.float32), {"method": "demucs", "vocals_path": str(wavs[0])}


def separate_vocals(
    y: np.ndarray,
    sr: int,
    audio_path: str | Path,
    *,
    max_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    보컬에 가까운 파형 반환 + 분리 메타.
    """
    use_demucs = os.environ.get("MUSIC_SEARCH_USE_DEMUCS", "").strip() in (
        "1",
        "true",
        "yes",
    )
    path = Path(audio_path)
    if use_demucs and path.is_file():
        cache = Path(__file__).resolve().parent.parent / "music_search_data" / "demucs_cache"
        v_dem, info = _separate_vocals_demucs_file(path, cache)
        if v_dem is not None and v_dem.size:
            # 길이 맞춤
            max_len = int(max_seconds * sr)
            if v_dem.shape[0] > max_len:
                v_dem = v_dem[:max_len]
            return v_dem, info

    return separate_vocals_hpss_bandpass(y, sr)
