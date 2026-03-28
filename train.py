"""
train.py  —  Hybrid Recommender  (scikit-learn only, no scikit-surprise)
========================================================================
Algorithm:
  CF  : TruncatedSVD on the mean-centred user×movie ratings matrix
  CB  : TF-IDF genre vectors + cosine similarity
  Save: model/hybrid_model.pkl

Usage:
    python train.py
    python train.py --factors 50
    python train.py --cf-weight 0.6
"""

import argparse, pickle, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("data/ml-100k")
MODEL_DIR  = Path("model")
MODEL_FILE = MODEL_DIR / "hybrid_model.pkl"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--factors",   type=int,   default=50,  help="SVD latent factors")
parser.add_argument("--cf-weight", type=float, default=0.6, help="CF blend weight 0-1")
args = parser.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Loading data...")
if not (DATA_PATH / "u.data").exists():
    print("[ERROR] Dataset not found. Run: python download_data.py")
    exit(1)

ratings = pd.read_csv(
    DATA_PATH / "u.data", sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)
mcols  = ["movie_id","title","release_date","video_date","url"] + [f"g{i}" for i in range(19)]
movies = pd.read_csv(DATA_PATH / "u.item", sep="|", names=mcols, encoding="latin-1")

GENRE_NAMES = ["unknown","Action","Adventure","Animation","Children's","Comedy",
               "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
               "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]

movies["genres"] = movies[[f"g{i}" for i in range(19)]].apply(
    lambda r: " ".join([GENRE_NAMES[i] for i, v in enumerate(r) if v == 1]), axis=1
)

print(f"    ✓ {len(ratings):,} ratings | "
      f"{ratings['movie_id'].nunique():,} movies | "
      f"{ratings['user_id'].nunique():,} users")

# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD USER×MOVIE MATRIX  (mean-centred per user)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Building user×movie matrix...")

user_ids  = sorted(ratings["user_id"].unique())
movie_ids = sorted(ratings["movie_id"].unique())
uidx      = {u: i for i, u in enumerate(user_ids)}
midx      = {m: i for i, m in enumerate(movie_ids)}

user_means = ratings.groupby("user_id")["rating"].mean()
ratings["rc"] = ratings.apply(
    lambda r: r["rating"] - user_means[r["user_id"]], axis=1
)

R = csr_matrix(
    (ratings["rc"].values,
     (ratings["user_id"].map(uidx).values,
      ratings["movie_id"].map(midx).values)),
    shape=(len(user_ids), len(movie_ids))
)

sparsity = 1 - len(ratings) / (R.shape[0] * R.shape[1])
print(f"    ✓ Matrix {R.shape}  |  sparsity {sparsity:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. COLLABORATIVE FILTERING  —  TruncatedSVD
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[3/5] Training TruncatedSVD (factors={args.factors})...")
t0  = time.time()
svd = TruncatedSVD(n_components=args.factors, random_state=42)
U   = svd.fit_transform(R)          # (n_users, factors)
Vt  = svd.components_               # (factors, n_movies)
R_hat = U @ Vt                      # full reconstructed matrix

print(f"    ✓ SVD done in {time.time()-t0:.1f}s  |  "
      f"explained variance {svd.explained_variance_ratio_.sum():.1%}")

# ── quick RMSE on observed entries ───────────────────────────────────────────
cx   = R.tocoo()
pred = np.array([R_hat[i, j] + user_means.get(user_ids[i], 0)
                 for i, j in zip(cx.row, cx.col)])
true = cx.data + np.array([user_means.get(user_ids[i], 0) for i in cx.row])
pred = np.clip(pred, 1, 5)
rmse = float(np.sqrt(mean_squared_error(true, pred)))
mae  = float(mean_absolute_error(true, pred))
print(f"    ✓ Train RMSE={rmse:.4f}  |  MAE={mae:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. CONTENT-BASED  —  TF-IDF + cosine similarity
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Building content-based model (TF-IDF genres)...")
movies_clean = movies[["movie_id","genres"]].dropna()
movies_clean = movies_clean[movies_clean["genres"].str.strip() != ""].copy()

tfidf       = TfidfVectorizer(token_pattern=r"[A-Za-z'\-]+")
tfidf_mat   = tfidf.fit_transform(movies_clean["genres"])
cb_matrix   = cosine_similarity(tfidf_mat)           # (N, N) numpy array
cb_movie_ids = movies_clean["movie_id"].tolist()

print(f"    ✓ CB matrix {cb_matrix.shape}  |  vocab {len(tfidf.vocabulary_)}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Saving model...")
MODEL_DIR.mkdir(exist_ok=True)

bundle = {
    # CF components
    "svd":        svd,
    "U":          U,
    "Vt":         Vt,
    "user_ids":   user_ids,
    "movie_ids":  movie_ids,    # ordered list for CF matrix columns
    "user_means": dict(user_means),
    # CB components
    "cb_matrix":  cb_matrix,
    "cb_movie_ids": cb_movie_ids,  # ordered list for CB matrix
    "tfidf":      tfidf,
    # Config
    "cf_weight":  args.cf_weight,
    "metrics":    {"rmse": round(rmse,4), "mae": round(mae,4)},
    "config":     {"n_factors": args.factors, "dataset": "MovieLens 100K"},
    "trained_on": pd.Timestamp.now().isoformat(),
}

with open(MODEL_FILE, "wb") as f:
    pickle.dump(bundle, f)

size_mb = MODEL_FILE.stat().st_size / 1024 / 1024
print(f"    ✓ Saved → {MODEL_FILE}  ({size_mb:.1f} MB)")

print("""
╔══════════════════════════════════════════════════╗
║  ✅  Training complete!                          ║
║                                                  ║
║  Launch the app:  streamlit run app.py           ║
╚══════════════════════════════════════════════════╝
""")
