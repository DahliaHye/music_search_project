"""
인덱스 구축 / 검색 파이프라인.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from . import config
from .audio_features import extract_all, l2_normalize
from .clustering import fit_clusters, save_cluster_report
from .filename_parse import guess_artist_title
from .hybrid_search import content_similarity_rank, hybrid_scores, load_meta
from .nlp_query import parse_natural_language
from .nn_projection import pca_projection
from .text_index import build_inverted_index, save_elasticsearch_like_bundle
from .textify import features_to_tags
from .vector_index import build_index_ip, save_faiss_index


def _audio_files() -> list[Path]:
    if not config.AUDIO_DIR.is_dir():
        return []
    return sorted(p for p in config.AUDIO_DIR.glob("*.mp3") if p.is_file())


def build_index(
    *,
    use_pca: bool = True,
    n_clusters: int | None = None,
) -> dict[str, Any]:
    paths = _audio_files()
    if not paths:
        raise FileNotFoundError(f"mp3 없음: {config.AUDIO_DIR}")

    combined_rows: list[np.ndarray] = []
    mix_rows: list[np.ndarray] = []
    vocal_rows: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []

    for p in paths:
        out = extract_all(str(p), max_seconds=config.MAX_AUDIO_SECONDS)
        combined_rows.append(out["vec_combined"])
        mix_rows.append(out["vec_mix"])
        vocal_rows.append(out["vec_vocal"])

        flat_for_tags = {**out["flat_mix"], **out["flat_vocal"]}
        tags = features_to_tags(flat_for_tags)
        ga, gt = guess_artist_title(p.name)
        meta.append(
            {
                "filename": p.name,
                "path": str(p.resolve()),
                "guess_artist": ga,
                "guess_title": gt,
                "text_tags": tags,
                "params": out["params"],
                "separation": out["separation"],
            }
        )
        param_rows.append(
            {
                "filename": p.name,
                **out["params"],
            }
        )

    X = np.stack(combined_rows, axis=0)
    X_mix = np.stack(mix_rows, axis=0)
    X_v = np.stack(vocal_rows, axis=0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca_path = config.DATA_DIR / "pca_model.pkl"
    if use_pca:
        Z, _pca = pca_projection(Xs, n_components=32, state_path=pca_path)
        emb_for_index = np.stack([l2_normalize(Z[i]) for i in range(Z.shape[0])])
    else:
        emb_for_index = np.stack([l2_normalize(Xs[i]) for i in range(Xs.shape[0])])

    v_scaler = StandardScaler()
    Xvs = v_scaler.fit_transform(X_v)
    Z_v, _ = pca_projection(
        Xvs,
        n_components=32,
        state_path=config.PCA_VOCAL_PATH,
    )
    vocal_emb = np.stack([l2_normalize(Z_v[i]) for i in range(Z_v.shape[0])])

    nc = n_clusters or config.DEFAULT_N_CLUSTERS
    labels, centroids, cmeta = fit_clusters(Xs, n_clusters=min(nc, len(paths)))
    for i, m in enumerate(meta):
        m["cluster_id"] = int(labels[i])

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        config.SCALER_PATH,
        mean=scaler.mean_,
        scale=scaler.scale_,
    )
    np.savez(
        config.VOCAL_SCALER_PATH,
        mean=v_scaler.mean_,
        scale=v_scaler.scale_,
    )
    np.save(config.EMB_PATH, emb_for_index)
    np.save(config.VOCAL_EMB_PATH, vocal_emb)
    np.save(config.RAW_MIX_PATH, X_mix)
    np.save(config.RAW_VOCAL_PATH, X_v)

    with open(config.META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(config.TRACK_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(param_rows, f, ensure_ascii=False, indent=2)

    index = build_index_ip(emb_for_index)
    save_faiss_index(index, config.INDEX_PATH)

    inv = build_inverted_index(meta)
    save_elasticsearch_like_bundle(config.DATA_DIR / "text_search", meta, inv)

    names = [m["filename"] for m in meta]
    save_cluster_report(
        config.CLUSTER_PATH,
        labels=labels,
        track_names=names,
        meta=cmeta,
    )

    return {
        "n_tracks": len(paths),
        "dim_embedding": emb_for_index.shape[1],
        "dim_vocal_embedding": vocal_emb.shape[1],
        "clusters": int(cmeta.get("n_clusters", 0)),
        "data_dir": str(config.DATA_DIR),
    }


def search_natural_language(
    query: str,
    *,
    seed_track_idx: int | None = None,
    top_k: int = 10,
    use_vocal_audio: bool = False,
) -> tuple[list[tuple[dict[str, Any], float]], dict[str, Any]]:
    if not config.META_PATH.is_file():
        raise FileNotFoundError("인덱스 없음. 먼저: python -m music_search build")
    meta = load_meta(config.META_PATH)
    emb = np.load(config.EMB_PATH)
    vocal_emb = np.load(config.VOCAL_EMB_PATH)
    descriptions = [m.get("text_tags", "") for m in meta]
    parsed = parse_natural_language(query)
    audio_alt = vocal_emb if use_vocal_audio else None
    ranked, dbg = hybrid_scores(
        parsed,
        embeddings=emb,
        descriptions=descriptions,
        track_meta=meta,
        seed_track_idx=seed_track_idx,
        audio_embeddings=audio_alt,
    )
    out: list[tuple[dict[str, Any], float]] = []
    for idx, score in ranked[:top_k]:
        out.append((meta[idx], score))
    return out, dbg


def similar_to_track(
    track_idx: int,
    top_k: int = 10,
    *,
    vocal: bool = False,
) -> list[tuple[dict[str, Any], float]]:
    meta = load_meta(config.META_PATH)
    emb = np.load(config.VOCAL_EMB_PATH if vocal else config.EMB_PATH)
    ranked = content_similarity_rank(emb, track_idx)
    return [(meta[i], s) for i, s in ranked[:top_k]]
