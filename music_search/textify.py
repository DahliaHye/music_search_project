"""
특징 수치를 짧은 영·한 혼합 태그 문자열로 바꿔 TF-IDF / 문장 임베딩 입력으로 씁니다.
"""
from __future__ import annotations

from typing import Any


def features_to_tags(flat: dict[str, Any]) -> str:
    """자연어 검색용 의사-문서 텍스트."""
    tempo = float(flat.get("tempo_bpm", 120))
    sc = float(flat.get("spectral_centroid_mean", 2000))
    zcr = float(flat.get("zcr_mean", 0.05))
    hp = float(flat.get("harm_perc_ratio", 1.0))

    if tempo < 95:
        pace = "slow tempo calm 느린 잔잔"
    elif tempo < 125:
        pace = "moderate tempo mid 중간"
    else:
        pace = "fast tempo upbeat 빠른 활기"

    if sc < 1800:
        tone = "dark warm low spectral 어두운 낮은"
    elif sc < 3500:
        tone = "balanced neutral 중성"
    else:
        tone = "bright airy high spectral 밝은 화사"

    if zcr > 0.12:
        nois = "noisy percussive 타격감"
    else:
        nois = "smooth harmonic 부드러운"

    if hp > 2.0:
        harm = "harmonic melodic 선율 위주"
    elif hp < 0.8:
        harm = "rhythmic percussive 리듬 위주"
    else:
        harm = "mixed harmonic percussive 혼합"

    rms = float(flat.get("rms_mean", 0.05))
    if rms < 0.03:
        dyn = "quiet soft 조용"
    elif rms < 0.1:
        dyn = "medium loudness 중간 볼륨"
    else:
        dyn = "loud energetic 강한 에너지"

    parts = [pace, tone, nois, harm, dyn, "music audio track"]

    if flat.get("vocal_rms_mean", 0) and float(flat.get("vocal_rms_mean", 0)) > 1e-6:
        vc = float(flat["vocal_spectral_centroid_mean"])
        vhf = float(flat.get("vocal_hf_energy_ratio", 0))
        if vc < 2000:
            vp = "vocal warm chesty 따뜻한 보컬"
        elif vc < 3800:
            vp = "vocal balanced natural 중성 보컬"
        else:
            vp = "vocal bright airy 밝은 고음 보컬"
        if vhf > 0.35:
            vp += " breathy airy 쉰 느낌"
        parts.append(vp)

    return " ".join(parts)
