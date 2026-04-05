"""
python -m music_search build
python -m music_search search "밝은 분위기 음악"
python -m music_search search "가수 Billie Eilish" --seed-idx 0 --vocal-audio
python -m music_search similar 0
python -m music_search similar 0 --vocal
python -m music_search viz
python -m music_search clusters
python -m music_search vocal-clusters
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .feature_map_viz import run_visualize
from .pipeline import build_index, search_natural_language, similar_to_track


def main() -> None:
    p = argparse.ArgumentParser(
        description="로컬 오디오 특징 인덱스 + 하이브리드 검색 데모",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="audio/*.mp3 로 인덱스 구축")
    b.add_argument(
        "--no-pca",
        action="store_true",
        help="PCA 축소 없이 스케일+L2 정규 특징 사용",
    )
    b.add_argument(
        "--clusters",
        type=int,
        default=None,
        help=f"KMeans 클러스터 수 — 결합 특징 (기본 {config.DEFAULT_N_CLUSTERS})",
    )
    b.add_argument(
        "--vocal-clusters",
        type=int,
        default=None,
        dest="vocal_clusters",
        help=f"보컬 특징만 쓰는 KMeans 클러스터 수 (기본 {config.DEFAULT_VOCAL_N_CLUSTERS})",
    )

    s = sub.add_parser("search", help="자연어 검색")
    s.add_argument("query", type=str, help="예: 잔잔한 분위기 / 가수 Rosé")
    s.add_argument(
        "--seed-idx",
        type=int,
        default=None,
        help="이 트랙 인덱스(0..n-1) 오디오와 혼합 유사도",
    )
    s.add_argument("-k", type=int, default=10, help="상위 k개")
    s.add_argument(
        "--vocal-audio",
        action="store_true",
        help="시드가 있을 때 보컬 전용 임베딩으로 오디오 유사도 계산",
    )

    sim = sub.add_parser("similar", help="트랙 인덱스 기준 순수 콘텐츠 유사도")
    sim.add_argument("track_idx", type=int)
    sim.add_argument("-k", type=int, default=10)
    sim.add_argument(
        "--vocal",
        action="store_true",
        help="보컬 전용 임베딩만으로 코사인 유사도",
    )

    vz = sub.add_parser("viz", help="FEATURE MAP PNG + 요약 JSON 저장")
    vz.add_argument(
        "--out",
        type=str,
        default=None,
        help="출력 폴더 (기본: music_search_data/figures)",
    )

    sub.add_parser("clusters", help="결합 특징 클러스터 요약 JSON 출력")
    sub.add_parser(
        "vocal-clusters",
        help="보컬 특징만 쓴 클러스터 요약 JSON 출력",
    )

    args = p.parse_args()

    if args.cmd == "build":
        info = build_index(
            use_pca=not args.no_pca,
            n_clusters=args.clusters,
            vocal_n_clusters=getattr(args, "vocal_clusters", None),
        )
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return

    if args.cmd == "search":
        rows, dbg = search_natural_language(
            args.query,
            seed_track_idx=args.seed_idx,
            top_k=args.k,
            use_vocal_audio=args.vocal_audio,
        )
        print("debug:", json.dumps(dbg, ensure_ascii=False))
        for m, sc in rows:
            vc = m.get("vocal_cluster_id")
            vc_s = f" v={vc}" if vc is not None else ""
            print(
                f"{sc:.4f}  [c={m.get('cluster_id')}]{vc_s}  {m.get('filename')}"
            )
            tt = m.get("text_tags", "") or ""
            tail = "..." if len(tt) > 160 else ""
            print(f"         tags: {tt[:160]}{tail}")
        return

    if args.cmd == "similar":
        rows = similar_to_track(
            args.track_idx,
            top_k=args.k,
            vocal=args.vocal,
        )
        for m, sc in rows:
            print(f"{sc:.4f}  {m.get('filename')}")
        return

    if args.cmd == "viz":
        out = Path(args.out) if args.out else None
        paths = run_visualize(out_dir=out)
        print(json.dumps(paths, ensure_ascii=False, indent=2))
        return

    if args.cmd == "clusters":
        path = config.CLUSTER_PATH
        if not path.is_file():
            print("클러스터 파일 없음. build 먼저 실행.", file=sys.stderr)
            sys.exit(1)
        print(path.read_text(encoding="utf-8"))
        return

    if args.cmd == "vocal-clusters":
        path = config.VOCAL_CLUSTER_PATH
        if not path.is_file():
            print("보컬 클러스터 파일 없음. build 먼저 실행.", file=sys.stderr)
            sys.exit(1)
        print(path.read_text(encoding="utf-8"))
        return


if __name__ == "__main__":
    main()
