"""
video 폴더(하위 포함)의 영상에서 오디오만 추출해 audio 폴더에 저장합니다.
곡 인식은 acrcloud_recognize.py → results.csv 를 사용하세요.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

BASE_DIR = Path(__file__).resolve().parent
VIDEO_ROOT = BASE_DIR / "video"
AUDIO_DIR = BASE_DIR / "audio"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")


def resolve_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    return path if path else imageio_ffmpeg.get_ffmpeg_exe()


def ensure_ffmpeg(ffmpeg: str) -> None:
    if not Path(ffmpeg).is_file():
        print(f"오류: ffmpeg 실행 파일을 찾을 수 없습니다.\n  → {ffmpeg}", file=sys.stderr)
        sys.exit(1)
    try:
        subprocess.run(
            [ffmpeg, "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"오류: ffmpeg를 실행할 수 없습니다.\n  → {ffmpeg}\n  {e}", file=sys.stderr)
        sys.exit(1)


def list_videos(folder: Path) -> list[Path]:
    """video 폴더 아래(하위 폴더 포함) 모든 영상."""
    if not folder.is_dir():
        print(f"오류: 영상 폴더가 없습니다.\n  → {folder}", file=sys.stderr)
        sys.exit(1)
    found: list[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            found.append(p)
    return sorted(found)


def audio_path_for_video(video_path: Path, video_root: Path, audio_dir: Path) -> Path:
    """하위 폴더에 같은 파일명이 있어도 겹치지 않도록 상대 경로를 파일명에 반영."""
    rel = video_path.relative_to(video_root)
    stem_key = rel.with_suffix("").as_posix().replace("/", "_").replace("\\", "_")
    return audio_dir / f"{stem_key}_audio.mp3"


def extract_audio(ffmpeg: str, video_path: Path, audio_path: Path) -> Path:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-q:a",
        "0",
        "-map",
        "a",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path


def main() -> None:
    ffmpeg = resolve_ffmpeg()
    ensure_ffmpeg(ffmpeg)
    print(f"ffmpeg: {ffmpeg}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    videos = list_videos(VIDEO_ROOT)
    if not videos:
        print(
            f"오류: 영상이 없습니다 ({', '.join(VIDEO_EXTENSIONS)}).\n"
            f"  → {VIDEO_ROOT} (하위 폴더 포함 검색)",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"처리할 영상: {len(videos)}개 → 오디오 저장: {AUDIO_DIR}")

    for video_path in videos:
        rel = str(video_path.relative_to(VIDEO_ROOT))
        audio_path = audio_path_for_video(video_path, VIDEO_ROOT, AUDIO_DIR)
        extract_audio(ffmpeg, video_path, audio_path)
        print(f"{rel}  →  {audio_path.relative_to(BASE_DIR)}")

    print(f"[완료] 오디오: {AUDIO_DIR}")
    print("       다음: python acrcloud_recognize.py  →  results.csv")


if __name__ == "__main__":
    main()
