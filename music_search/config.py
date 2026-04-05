from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "audio"
DATA_DIR = BASE_DIR / "music_search_data"
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "tracks_meta.json"
EMB_PATH = DATA_DIR / "embeddings.npy"
VOCAL_EMB_PATH = DATA_DIR / "vocal_embeddings.npy"
SCALER_PATH = DATA_DIR / "feature_scaler.npz"
VOCAL_SCALER_PATH = DATA_DIR / "vocal_feature_scaler.npz"
CLUSTER_PATH = DATA_DIR / "clusters.json"
VOCAL_CLUSTER_PATH = DATA_DIR / "vocal_clusters.json"
RAW_MIX_PATH = DATA_DIR / "raw_mix_features.npy"
RAW_VOCAL_PATH = DATA_DIR / "raw_vocal_features.npy"
TRACK_PARAMS_PATH = DATA_DIR / "track_params.json"
FIGURES_DIR = DATA_DIR / "figures"
PCA_VOCAL_PATH = DATA_DIR / "pca_vocal.pkl"

# 인덱싱 시 각 트랙 앞에서 잘라 분석 (초) — 길이·속도 절충
MAX_AUDIO_SECONDS = 90.0

# KMeans 클러스터 수 (결합 특징 = 믹스+보컬 / 보컬 전용)
DEFAULT_N_CLUSTERS = 6
DEFAULT_VOCAL_N_CLUSTERS = 6
