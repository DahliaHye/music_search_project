"""
특징 패턴·도출 파라미터 시각화 (FEATURE MAP).
- 결합 임베딩 PCA 2D + 결합 클러스터 색
- 동일 PCA 좌표 + 보컬 전용 클러스터 색 (tracks_meta 에 vocal_cluster_id 있을 때)
- 트랙×파라미터 히트맵
- 클러스터별 레이더(결합 / 보컬 각각)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def run_visualize(
    out_dir: Path | None = None,
    *,
    random_state: int = 42,
) -> dict[str, str]:
    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA

    from . import config

    out_dir = out_dir or config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not config.EMB_PATH.is_file():
        raise FileNotFoundError("인덱스 없음. python -m music_search build")

    emb = np.load(config.EMB_PATH)
    meta = json.loads(config.META_PATH.read_text(encoding="utf-8"))
    params_list = []
    if config.TRACK_PARAMS_PATH.is_file():
        params_list = json.loads(
            config.TRACK_PARAMS_PATH.read_text(encoding="utf-8")
        )

    labels = np.array([m.get("cluster_id", 0) for m in meta], dtype=int)
    names = [m.get("filename", "?") for m in meta]

    # --- 1) PCA 2D feature map
    pca2 = PCA(n_components=2, random_state=random_state)
    xy = pca2.fit_transform(emb)
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.85,
        s=42,
        edgecolors="k",
        linewidths=0.2,
    )
    ax.set_title("FEATURE MAP: embedding PCA-2D (color = cluster)")
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
    plt.colorbar(sc, ax=ax, label="cluster_id")
    # 파일명은 한글 폰트 이슈가 있어 인덱스만 표기 (파일명은 tracks_meta.json 참고)
    for i in range(len(names)):
        ax.annotate(
            str(i),
            (xy[i, 0], xy[i, 1]),
            fontsize=6,
            alpha=0.8,
        )
    p1 = out_dir / "feature_map_pca2d.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    p1v: Path | None = None
    v_lab = [
        m.get("vocal_cluster_id") for m in meta
    ]
    if v_lab and all(x is not None for x in v_lab):
        v_labels = np.array([int(x) for x in v_lab], dtype=int)
        fig, ax = plt.subplots(figsize=(10, 7))
        scv = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=v_labels,
            cmap="tab10",
            alpha=0.85,
            s=42,
            edgecolors="k",
            linewidths=0.2,
        )
        ax.set_title(
            "FEATURE MAP: same PCA-2D, color = vocal-only cluster"
        )
        ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
        plt.colorbar(scv, ax=ax, label="vocal_cluster_id")
        for i in range(len(names)):
            ax.annotate(str(i), (xy[i, 0], xy[i, 1]), fontsize=6, alpha=0.8)
        p1v = out_dir / "feature_map_pca2d_vocal_clusters.png"
        fig.tight_layout()
        fig.savefig(p1v, dpi=160)
        plt.close(fig)

    # --- 2) 파라미터 히트맵 (mood + voice)
    mood_keys = ["energy", "brightness", "tempo_norm", "harmonic_emphasis"]
    voice_keys = [
        "vocal_brightness",
        "vocal_presence",
        "vocal_air_hf",
        "vocal_prominence_vs_mix",
    ]
    dims = mood_keys + voice_keys
    rows: list[list[float]] = []
    row_labels: list[str] = []
    if params_list:
        for i, pr in enumerate(params_list):
            row_labels.append(f"#{i}")
            m = pr.get("mood", {})
            v = pr.get("voice", {})
            rows.append(
                [float(m.get(k, 0)) for k in mood_keys]
                + [float(v.get(k, 0)) for k in voice_keys]
            )
    p2: Path | None = None
    if rows:
        M = np.array(rows, dtype=np.float64)
        fig, ax = plt.subplots(figsize=(12, max(6, len(rows) * 0.22)))
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(mood_keys) + len(voice_keys)))
        ax.set_xticklabels(mood_keys + voice_keys, rotation=35, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title("FEATURE MAP: mood & voice parameters (0–1)")
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        p2 = out_dir / "feature_map_param_heatmap.png"
        fig.tight_layout()
        fig.savefig(p2, dpi=160)
        plt.close(fig)

    # --- 3) 클러스터 레이더 (파라미터 평균)
    p3: Path | None = None
    if params_list:
        n_c = int(labels.max()) + 1 if labels.size else 0
        cent = np.zeros((n_c, len(dims)), dtype=np.float64)
        counts = np.zeros(n_c, dtype=np.int32)
        for i, pr in enumerate(params_list):
            cid = int(labels[i]) if i < len(labels) else 0
            m = pr.get("mood", {})
            v = pr.get("voice", {})
            vec = [float(m.get(k, 0)) for k in mood_keys] + [
                float(v.get(k, 0)) for k in voice_keys
            ]
            cent[cid] += np.array(vec)
            counts[cid] += 1
        for c in range(n_c):
            if counts[c] > 0:
                cent[c] /= counts[c]

        angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, polar=True)
        for c in range(n_c):
            if counts[c] == 0:
                continue
            vals = np.concatenate([cent[c], [cent[c, 0]]])
            ax.plot(angles, vals, label=f"cluster {c} (n={counts[c]})")
            ax.fill(angles, vals, alpha=0.08)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dims, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title("Cluster radar: mean mood & voice params")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
        p3 = out_dir / "feature_map_cluster_radar.png"
        fig.tight_layout()
        fig.savefig(p3, dpi=160)
        plt.close(fig)

    # --- 3b) 보컬 클러스터 레이더 (동일 파라미터, 그룹만 보컬 ID)
    p3v: Path | None = None
    if (
        params_list
        and v_lab
        and all(x is not None for x in v_lab)
    ):
        v_labels_r = np.array([int(x) for x in v_lab], dtype=int)
        n_vc = int(v_labels_r.max()) + 1 if v_labels_r.size else 0
        cent_v = np.zeros((n_vc, len(dims)), dtype=np.float64)
        counts_v = np.zeros(n_vc, dtype=np.int32)
        for i, pr in enumerate(params_list):
            cid = int(v_labels_r[i]) if i < len(v_labels_r) else 0
            m = pr.get("mood", {})
            v = pr.get("voice", {})
            vec = [float(m.get(k, 0)) for k in mood_keys] + [
                float(v.get(k, 0)) for k in voice_keys
            ]
            cent_v[cid] += np.array(vec)
            counts_v[cid] += 1
        for c in range(n_vc):
            if counts_v[c] > 0:
                cent_v[c] /= counts_v[c]

        angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, polar=True)
        for c in range(n_vc):
            if counts_v[c] == 0:
                continue
            vals = np.concatenate([cent_v[c], [cent_v[c, 0]]])
            ax.plot(angles, vals, label=f"vocal cluster {c} (n={counts_v[c]})")
            ax.fill(angles, vals, alpha=0.08)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dims, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title("Vocal-cluster radar: mean mood & voice params")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
        p3v = out_dir / "feature_map_vocal_cluster_radar.png"
        fig.tight_layout()
        fig.savefig(p3v, dpi=160)
        plt.close(fig)

    # 요약 JSON
    summary = {
        "track_index_to_filename": {str(i): names[i] for i in range(len(names))},
        "pca2_explained_ratio": pca2.explained_variance_ratio_.tolist(),
        "n_tracks": int(emb.shape[0]),
        "output_files": {
            "pca2d": str(p1),
            "pca2d_vocal_clusters": str(p1v) if p1v else None,
            "heatmap": str(p2) if p2 else None,
            "radar_combined": str(p3) if p3 else None,
            "radar_vocal_clusters": str(p3v) if p3v else None,
        },
    }
    summary_path = out_dir / "feature_map_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "pca2d": str(p1),
        "pca2d_vocal_clusters": str(p1v) if p1v else "",
        "heatmap": str(p2) if p2 else "",
        "radar_combined": str(p3) if p3 else "",
        "radar_vocal_clusters": str(p3v) if p3v else "",
        "summary_json": str(summary_path),
    }
