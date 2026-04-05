"""
ACRCloud로 audio 폴더의 mp3에서 곡 제목·가수 조회 → results.csv 저장.

네이티브 SDK 없이 Identification API 의 data_type=audio 로 요청합니다.
  https://docs.acrcloud.com/reference/identification-api/identification-api

자격 증명:
  acrcloud_credentials.py (HOST, ACCESS_KEY, ACCESS_SECRET) 또는
  환경 변수 ACRCLOUD_HOST, ACRCLOUD_ACCESS_KEY, ACRCLOUD_ACCESS_SECRET

대상: audio/*.mp3 (main.py 로 추출한 파일 등)

파일명 규칙(참고): 가수_곡_audio.mp3
  예: BTS_SWIM_audio.mp3 → 가수 BTS, 곡 SWIM
  첫 번째 _ 만 구분선으로 쓰고, 곡 제목 안에 _ 가 있어도 됩니다.

results.csv: 열 2개 — 파일명, 인식 결과(괄호 안에 파일명 해석·비고 등).
"""
from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import requests

BASE_DIR = Path(__file__).resolve().parent
AUDIO_FOLDER = BASE_DIR / "audio"
OUTPUT_CSV = BASE_DIR / "results.csv"
CLIP_SECONDS = 12


def _configure_windows_console() -> None:
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _safe_print(*parts: object) -> None:
    line = " ".join(str(p) for p in parts)
    try:
        print(line)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(line.encode(enc, errors="replace").decode(enc, errors="replace"))


def load_credentials() -> tuple[str, str, str]:
    host = os.environ.get("ACRCLOUD_HOST", "").strip()
    key = os.environ.get("ACRCLOUD_ACCESS_KEY", "").strip()
    secret = os.environ.get("ACRCLOUD_ACCESS_SECRET", "").strip()
    if host and key and secret:
        return host, key, secret
    try:
        import acrcloud_credentials as cred

        return (
            cred.HOST.strip(),
            cred.ACCESS_KEY.strip(),
            cred.ACCESS_SECRET.strip(),
        )
    except ImportError:
        pass
    return "", "", ""


def resolve_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    return path if path else imageio_ffmpeg.get_ffmpeg_exe()


def audio_clip_bytes(ffmpeg: str, path: Path) -> tuple[bytes, str]:
    """인식용으로 앞부분만 잘라 용량·시간 제한에 맞춤."""
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-t",
        str(CLIP_SECONDS),
        "-acodec",
        "libmp3lame",
        "-q:a",
        "4",
        "-f",
        "mp3",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode == 0 and proc.stdout:
        return proc.stdout, "audio/mpeg"
    data = path.read_bytes()
    if len(data) > 900_000:
        data = data[:900_000]
    return data, "audio/mpeg"


def identify_by_audio(
    host: str,
    access_key: str,
    access_secret: str,
    audio_bytes: bytes,
    filename: str,
    mime: str,
    timeout: int = 30,
) -> str:
    http_method = "POST"
    http_uri = "/v1/identify"
    data_type = "audio"
    signature_version = "1"
    timestamp = int(time.time())

    string_to_sign = (
        f"{http_method}\n{http_uri}\n{access_key}\n{data_type}\n"
        f"{signature_version}\n{timestamp}"
    )
    sign = base64.b64encode(
        hmac.new(
            access_secret.encode("ascii"),
            string_to_sign.encode("ascii"),
            digestmod=hashlib.sha1,
        ).digest()
    ).decode("ascii")

    url = f"https://{host.rstrip('/')}{http_uri}"
    fields = {
        "access_key": access_key,
        "sample_bytes": str(len(audio_bytes)),
        "timestamp": str(timestamp),
        "signature": sign,
        "data_type": data_type,
        "signature_version": signature_version,
    }
    files = {"sample": (filename, audio_bytes, mime)}
    resp = requests.post(url, data=fields, files=files, timeout=timeout)
    resp.encoding = "utf-8"
    return resp.text


def parse_music_line(raw: str) -> tuple[str, str, str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ("", "", raw[:800])

    status = data.get("status") or {}
    code = status.get("code")
    if code is not None and int(code) != 0:
        return ("", "", (status.get("msg") or str(data))[:800])

    meta = data.get("metadata") or {}
    music_list = meta.get("music") or []
    if not music_list:
        return ("", "", json.dumps(data, ensure_ascii=False)[:800])

    m0 = music_list[0]
    title = m0.get("title") or ""
    artists = m0.get("artists") or []
    artist_str = ", ".join(
        a.get("name", "") for a in artists if isinstance(a, dict)
    )
    return (title, artist_str, "")


def list_audio_mp3s(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".mp3")


def stem_without_audio_suffix(filename: str) -> str:
    """1_audio.mp3 / 가수_곡_audio.mp3 → stem 에서 _audio 제거."""
    stem = Path(filename).stem
    if stem.endswith("_audio"):
        return stem[: -len("_audio")]
    return stem


def parse_expected_artist_title(filename: str) -> tuple[str, str]:
    """
    가수_곡 형식: 첫 _ 기준 분리.
    _ 없으면 곡만 있다고 보고 (가수 빈칸, 전체를 곡 후보로).
    """
    base = stem_without_audio_suffix(filename)
    if "_" not in base:
        return ("", base.strip())
    artist, title = base.split("_", 1)
    return (artist.strip(), title.strip())


def format_csv_result_cell(
    title: str,
    artists: str,
    exp_a: str,
    exp_t: str,
    note: str,
) -> str:
    """앞: 인식 제목·가수, 뒤: 괄호 안 부가 정보."""
    if title or artists:
        main = f"{title or '—'} | {artists or '—'}"
    else:
        main = "(인식 없음)"
    extra = f"파일명 해석: 가수={exp_a or '—'} · 곡={exp_t or '—'}"
    if note:
        extra = f"{extra} · 비고: {note[:500]}"
    return f"{main} ({extra})"


def main() -> None:
    _configure_windows_console()

    host, key, secret = load_credentials()
    if not host or not key or not secret:
        print(
            "자격 증명이 없습니다.\n"
            "  acrcloud_credentials.py 또는 환경 변수를 설정하세요.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not AUDIO_FOLDER.is_dir():
        print(f"폴더 없음: {AUDIO_FOLDER}", file=sys.stderr)
        sys.exit(1)

    audio_files = list_audio_mp3s(AUDIO_FOLDER)
    if not audio_files:
        print(
            f"{AUDIO_FOLDER} 에 .mp3 파일이 없습니다. 먼저 main.py 로 음원을 추출하세요.",
            file=sys.stderr,
        )
        sys.exit(1)

    ffmpeg = resolve_ffmpeg()
    header = ["파일명", "인식 결과 (괄호: 파일명 해석·비고)"]
    rows: list[list[str]] = []
    for audio_path in audio_files:
        clip, mime = audio_clip_bytes(ffmpeg, audio_path)
        raw = identify_by_audio(host, key, secret, clip, audio_path.name, mime)
        title, artists, note = parse_music_line(raw)
        exp_a, exp_t = parse_expected_artist_title(audio_path.name)
        cell = format_csv_result_cell(title, artists, exp_a, exp_t, note)
        rows.append([audio_path.name, cell])
        _safe_print(audio_path.name)
        _safe_print("  인식:", title or "(없음)", "|", artists or "(없음)")
        _safe_print(
            "  파일명 해석:",
            exp_a or "(가수 없음)",
            "/",
            exp_t,
        )
        if note:
            _safe_print("  비고:", note[:200])
        _safe_print("-" * 40)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    _safe_print(f"[완료] {OUTPUT_CSV} ({len(rows)}건)")


if __name__ == "__main__":
    main()
