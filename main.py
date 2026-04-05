"""
video 폴더(하위 포함)의 영상에서 오디오만 추출해 audio 폴더에 저장합니다.
곡 인식은 acrcloud_recognize.py → results.csv 를 사용하세요.

- 현재 영상 목록에 해당하지 않는 audio/*.mp3(이전에 남은 파일)는 실행 시 삭제합니다.
- 대상 mp3가 이미 있고, 영상 파일보다 새거나 같으면 ffmpeg로 다시 뽑지 않습니다.
  (영상을 수정·교체하면 수정 시각이 바뀌어 다시 추출됩니다.)
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


def expected_audio_paths(
    videos: list[Path], video_root: Path, audio_dir: Path
) -> set[Path]:
    return {
        audio_path_for_video(v, video_root, audio_dir).resolve() for v in videos
    }


def remove_orphan_mp3s(audio_dir: Path, expected_resolved: set[Path]) -> int:
    """영상에 대응하지 않는 audio/*.mp3 삭제. 삭제한 개수 반환."""
    if not audio_dir.is_dir():
        return 0
    removed = 0
    for p in audio_dir.glob("*.mp3"):
        if not p.is_file():
            continue
        if p.resolve() not in expected_resolved:
            p.unlink()
            removed += 1
    return removed


def needs_audio_extract(video_path: Path, audio_path: Path) -> bool:
    """오디오가 없거나 영상이 더 최근이면 True."""
    if not audio_path.is_file():
        return True
    try:
        v_mtime = video_path.stat().st_mtime
        a_mtime = audio_path.stat().st_mtime
    except OSError:
        return True
    return v_mtime > a_mtime


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

    expected = expected_audio_paths(videos, VIDEO_ROOT, AUDIO_DIR)
    n_orphan = remove_orphan_mp3s(AUDIO_DIR, expected)
    if n_orphan:
        print(f"영상에 없는 오디오 {n_orphan}개 삭제 (이전 실행 잔여)")

    n_extract = 0
    n_skip = 0
    for video_path in videos:
        rel = str(video_path.relative_to(VIDEO_ROOT))
        audio_path = audio_path_for_video(video_path, VIDEO_ROOT, AUDIO_DIR)
        if not needs_audio_extract(video_path, audio_path):
            print(f"{rel}  →  건너뜀 (이미 최신): {audio_path.relative_to(BASE_DIR)}")
            n_skip += 1
            continue
        extract_audio(ffmpeg, video_path, audio_path)
        n_extract += 1
        print(f"{rel}  →  {audio_path.relative_to(BASE_DIR)}")

    print(f"[완료] 오디오: {AUDIO_DIR}  (새로 추출 {n_extract}개, 건너뜀 {n_skip}개)")
    print("       다음: python acrcloud_recognize.py  →  results.csv")


if __name__ == "__main__":
    main()
