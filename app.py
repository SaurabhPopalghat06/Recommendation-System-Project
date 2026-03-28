import streamlit as st
import pandas as pd
import numpy as np
import pickle
import subprocess
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-SETUP  (runs once on Streamlit Cloud cold start)
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH  = Path("data/ml-100k")
MODEL_PATH = Path("model/hybrid_model.pkl")

def run_setup():
    if not (DATA_PATH / "u.data").exists():
        with st.spinner("⬇️ Downloading MovieLens 100K dataset (~5 MB)..."):
            r = subprocess.run([sys.executable, "download_data.py"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                st.error(f"Download failed:\n{r.stderr}")
                st.stop()

    if not MODEL_PATH.exists():
        with st.spinner("🧠 Training hybrid model — SVD + TF-IDF (~30 sec)..."):
            r = subprocess.run([sys.executable, "train.py"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                st.error(f"Training failed:\n{r.stderr}")
                st.stop()
        st.success("✅ Setup complete! Loading app...")
        st.rerun()

run_setup()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CineHybrid Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
        padding:2rem; border-radius:12px; margin-bottom:2rem;
        text-align:center; color:white;
    }
    .main-header h1{font-size:2.5rem;margin:0}
    .main-header p {font-size:1rem;opacity:.8;margin-top:.5rem}
    .rec-card{
        background:white; border-radius:10px; padding:1rem;
        border:1px solid #e0e0e0; margin-bottom:.5rem;
    }
    .score-badge{
        display:inline-block; padding:2px 10px;
        border-radius:20px; font-size:.75rem; font-weight:bold;
    }
    .badge-hybrid{background:#e8f4f8;color:#0f3460}
    .badge-cf    {background:#e8f8e8;color:#1a6b1a}
    .badge-cb    {background:#f8f0e8;color:#8b4500}
    .stButton>button{border-radius:8px;font-weight:600}
    div[data-testid="stSidebarContent"]{background:#1a1a2e}
    div[data-testid="stSidebarContent"] .stMarkdown p{color:#ccc}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    if not (DATA_PATH / "u.data").exists():
        return None, None, None
    ratings = pd.read_csv(DATA_PATH/"u.data", sep="\t",
                          names=["user_id","movie_id","rating","timestamp"])
    mcols   = ["movie_id","title","release_date","video_date","url"] + [f"g{i}" for i in range(19)]
    movies  = pd.read_csv(DATA_PATH/"u.item", sep="|", names=mcols, encoding="latin-1")
    GENRES  = ["unknown","Action","Adventure","Animation","Children's","Comedy",
               "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
               "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]
    movies["genres"] = movies[[f"g{i}" for i in range(19)]].apply(
        lambda r: "|".join([GENRES[i] for i, v in enumerate(r) if v == 1]), axis=1
    )
    movies["year"] = movies["title"].str.extract(r'\((\d{4})\)').fillna("")
    return ratings, movies, GENRES

# ══════════════════════════════════════════════════════════════════════════════
# HYBRID RECOMMENDATION ENGINE  (pure sklearn / numpy)
# ══════════════════════════════════════════════════════════════════════════════
def get_recommendations(model, user_ratings: dict, movies_df, top_n=10):
    if model is None or not user_ratings:
        return pd.DataFrame()

    # ── unpack model ──────────────────────────────────────────────────────────
    U          = model["U"]            # (n_users, factors)
    Vt         = model["Vt"]           # (factors, n_movies)
    movie_ids  = model["movie_ids"]    # CF movie ordering
    user_means = model["user_means"]   # {user_id: mean_rating}
    cb_matrix  = model["cb_matrix"]   # (n_cb, n_cb) cosine sim
    cb_ids     = model["cb_movie_ids"] # CB movie ordering
    cf_weight  = model.get("cf_weight", 0.6)
    cb_weight  = 1.0 - cf_weight

    # ── CF: build a pseudo-user vector from ratings ───────────────────────────
    # Project the user's known ratings into the SVD latent space
    global_mean  = float(np.mean(list(user_means.values())))
    pseudo_mean  = float(np.mean(list(user_ratings.values())))
    mid_to_cf    = {m: i for i, m in enumerate(movie_ids)}
    cb_mid_to_i  = {m: i for i, m in enumerate(cb_ids)}

    # Build a centred rating vector for the pseudo-user
    n_movies   = len(movie_ids)
    user_vec   = np.zeros(n_movies)
    for mid, r in user_ratings.items():
        if mid in mid_to_cf:
            user_vec[mid_to_cf[mid]] = r - pseudo_mean

    # Project into latent space then reconstruct scores
    latent        = user_vec @ Vt.T          # (factors,)
    cf_scores_all = latent @ Vt              # (n_movies,)  centred predictions

    rated_set = set(user_ratings.keys())
    scores    = {}

    for mid in movie_ids:
        if mid in rated_set:
            continue
        ci = mid_to_cf[mid]

        # CF score (add back pseudo-user mean + clamp to 1-5)
        cf_score = float(np.clip(cf_scores_all[ci] + pseudo_mean, 1.0, 5.0))

        # CB score: weighted avg similarity to rated movies (weight = normalised rating)
        cb_score = 0.0
        if mid in cb_mid_to_i:
            cidx       = cb_mid_to_i[mid]
            sim_vals   = []
            for rated_id, rating in user_ratings.items():
                if rated_id in cb_mid_to_i:
                    ridx = cb_mid_to_i[rated_id]
                    sim  = float(cb_matrix[cidx, ridx])
                    sim_vals.append(sim * (rating / 5.0))
            if sim_vals:
                cb_score = float(np.mean(sim_vals)) * 5.0

        hybrid = cf_weight * cf_score + cb_weight * cb_score
        scores[mid] = {"cf": round(cf_score, 2),
                       "cb": round(cb_score, 2),
                       "hybrid": round(hybrid, 2)}

    if not scores:
        return pd.DataFrame()

    result = (pd.DataFrame(scores).T
                .reset_index().rename(columns={"index": "movie_id"})
                .sort_values("hybrid", ascending=False)
                .head(top_n))
    result = result.merge(
        movies_df[["movie_id","title","genres","year"]], on="movie_id", how="left"
    )
    return result

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
model                            = load_model()
ratings_df, movies_df, GENRES   = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 CineHybrid")
    st.markdown("---")
    page = st.radio("Navigate",
        ["🏠 Home","⭐ Rate & Recommend","🔍 Search & Explore",
         "📊 Model Metrics","🗺️ Similarity Maps"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    if model:
        st.success("✅ Model loaded")
        st.caption(f"Trained: {model.get('trained_on','')[:10]}")
    else:
        st.warning("⚠️ Model not found.")
    if ratings_df is not None:
        st.info(f"📦 {len(ratings_df):,} ratings")
    else:
        st.error("❌ Dataset missing.")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🎬 CineHybrid Recommender</h1>
  <p>Hybrid AI · Collaborative Filtering + Content-Based · MovieLens 100K</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    if ratings_df is not None:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🎬 Movies",     f"{movies_df['movie_id'].nunique():,}")
        c2.metric("👥 Users",      f"{ratings_df['user_id'].nunique():,}")
        c3.metric("⭐ Ratings",    f"{len(ratings_df):,}")
        c4.metric("📊 Avg Rating", f"{ratings_df['rating'].mean():.2f} / 5")

    st.markdown("### 🧠 How the Hybrid System Works")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""#### 👥 Collaborative Filtering
Uses **TruncatedSVD** on the user×movie matrix to learn latent preference
factors. Projects your ratings into this space to predict scores for unseen movies.""")
    with c2:
        st.markdown("""#### 🎭 Content-Based Filtering
Builds a **TF-IDF genre vector** for every movie and computes pairwise
cosine similarity. Recommends movies genre-similar to your highest-rated ones.""")
    with c3:
        st.markdown("""#### 🔀 Hybrid Blending
`score = 0.6 × CF + 0.4 × CB` by default. Adjustable live via slider.
Solves cold-start — CB kicks in when CF has little signal.""")

    st.markdown("### 🚀 Local Setup")
    st.code("""pip install -r requirements.txt
python download_data.py
python train.py
streamlit run app.py""", language="bash")
    st.info("☁️ **Streamlit Cloud:** first load auto-downloads data and trains the model.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RATE & RECOMMEND
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⭐ Rate & Recommend":
    st.markdown("## ⭐ Rate Movies & Get Recommendations")
    if movies_df is None:
        st.error("Dataset not found."); st.stop()

    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 🎬 Rate Movies")
        gf = st.multiselect("Filter by genre", GENRES[1:], default=["Action","Drama","Comedy"])
        if gf:
            mask = movies_df["genres"].apply(lambda g: any(x in g for x in gf))
            pool = movies_df[mask].sample(min(20, int(mask.sum())), random_state=42)
        else:
            pool = movies_df.sample(20, random_state=42)

        for _, row in pool.iterrows():
            mid = int(row["movie_id"])
            a, b = st.columns([3,2])
            with a:
                st.write(f"**{row['title']}**")
                st.caption(str(row["genres"])[:40])
            with b:
                val = st.session_state.user_ratings.get(mid, 0)
                r   = st.select_slider("", [0,1,2,3,4,5], value=val,
                                        key=f"r_{mid}", label_visibility="collapsed")
                if r > 0:
                    st.session_state.user_ratings[mid] = r
                elif mid in st.session_state.user_ratings:
                    del st.session_state.user_ratings[mid]
            st.divider()

    with col_r:
        st.markdown("### 🎯 Your Recommendations")
        rated = st.session_state.user_ratings
        st.info(f"You have rated **{len(rated)}** movie(s).")

        if len(rated) < 2:
            st.warning("Rate at least **2 movies** to get recommendations.")
        elif model is None:
            st.error("Model not available.")
        else:
            top_n = st.slider("How many recommendations?", 5, 20, 10)
            cf_w  = st.slider("CF weight  (rest goes to Content-Based)", 0.0, 1.0, 0.6, 0.05)
            model["cf_weight"] = cf_w

            with st.spinner("Computing hybrid scores..."):
                recs = get_recommendations(model, rated, movies_df, top_n)

            if recs.empty:
                st.warning("No recommendations found.")
            else:
                for _, row in recs.iterrows():
                    hp = int(row["hybrid"] / 5 * 100)
                    cp = int(row["cf"]     / 5 * 100)
                    bp = int(row["cb"]     / 5 * 100)
                    st.markdown(f"""
                    <div class="rec-card">
                      <b>{row['title']}</b>
                      <span style="color:#888;font-size:.8rem"> · {str(row.get('genres',''))[:35]}</span><br>
                      <span class="score-badge badge-hybrid">🔀 Hybrid {hp}%</span>&nbsp;
                      <span class="score-badge badge-cf">👥 CF {cp}%</span>&nbsp;
                      <span class="score-badge badge-cb">🎭 CB {bp}%</span>
                      <div style="background:#eee;border-radius:4px;height:6px;margin-top:6px">
                        <div style="background:#0f3460;width:{hp}%;height:6px;border-radius:4px"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SEARCH & EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Search & Explore":
    st.markdown("## 🔍 Search & Explore Movies")
    if movies_df is None:
        st.error("Dataset not found."); st.stop()

    import plotly.express as px

    query = st.text_input("🔎 Search by title or genre", placeholder="e.g. Star Wars, Comedy...")
    c1, c2 = st.columns(2)
    with c1: gs  = st.multiselect("Filter by genre", GENRES[1:])
    with c2: sb  = st.selectbox("Sort by", ["Title A-Z","Year (newest)","Avg Rating","# Ratings"])

    df = movies_df.copy()
    if query:
        df = df[df["title"].str.contains(query, case=False, na=False) |
                df["genres"].str.contains(query, case=False, na=False)]
    if gs:
        df = df[df["genres"].apply(lambda g: any(x in g for x in gs))]

    if ratings_df is not None:
        stats = ratings_df.groupby("movie_id").agg(
            avg_rating=("rating","mean"), num_ratings=("rating","count")).reset_index()
        df = df.merge(stats, on="movie_id", how="left")
        df["avg_rating"]  = df["avg_rating"].fillna(0).round(2)
        df["num_ratings"] = df["num_ratings"].fillna(0).astype(int)

    if   sb == "Title A-Z":      df = df.sort_values("title")
    elif sb == "Year (newest)":  df = df.sort_values("year", ascending=False)
    elif sb == "Avg Rating"   and "avg_rating"  in df: df = df.sort_values("avg_rating",  ascending=False)
    elif sb == "# Ratings"    and "num_ratings" in df: df = df.sort_values("num_ratings", ascending=False)

    st.markdown(f"**{len(df):,} movies found**")
    cols = ["title","genres","year"] + (["avg_rating","num_ratings"] if "avg_rating" in df.columns else [])
    st.dataframe(df[cols].head(100).rename(columns={
        "title":"Title","genres":"Genres","year":"Year",
        "avg_rating":"Avg ⭐","num_ratings":"# Ratings"}),
        use_container_width=True, height=480)

    st.markdown("### 📊 Genre Distribution")
    gc = {}
    for g in movies_df["genres"]:
        for p in str(g).split("|"):
            if p and p != "unknown": gc[p] = gc.get(p,0)+1
    gdf = pd.DataFrame(list(gc.items()), columns=["Genre","Count"]).sort_values("Count")
    fig = px.bar(gdf, x="Count", y="Genre", orientation="h",
                 color="Count", color_continuous_scale="Blues", title="Movies per Genre")
    fig.update_layout(height=480, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Metrics":
    st.markdown("## 📊 Model Accuracy & Evaluation")
    if model is None:
        st.error("No trained model found."); st.stop()

    import plotly.express as px
    import plotly.graph_objects as go

    m   = model.get("metrics", {})
    cfg = model.get("config",  {})

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("RMSE",           f"{m.get('rmse',0):.4f}")
    c2.metric("MAE",            f"{m.get('mae',0):.4f}")
    c3.metric("Latent Factors", cfg.get("n_factors","—"))
    c4.metric("Movies Indexed", len(model.get("movie_ids",[])))

    if ratings_df is not None:
        st.markdown("### 📈 Rating Distribution")
        dist = ratings_df["rating"].value_counts().sort_index().reset_index()
        dist.columns = ["Rating","Count"]
        fig = px.bar(dist, x="Rating", y="Count", color="Count",
                     color_continuous_scale="Blues", title="Rating Distribution (1–5)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📐 SVD Explained Variance")
    svd = model.get("svd")
    if svd is not None:
        ev  = svd.explained_variance_ratio_
        evdf = pd.DataFrame({"Component": range(1, len(ev)+1),
                             "Variance": ev,
                             "Cumulative": np.cumsum(ev)})
        fig2 = px.line(evdf, x="Component", y="Cumulative", markers=False,
                       title="Cumulative Explained Variance by SVD Component")
        fig2.add_bar(x=evdf["Component"], y=evdf["Variance"], name="Per component",
                     marker_color="lightblue")
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SIMILARITY MAPS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Similarity Maps":
    st.markdown("## 🗺️ Movie Similarity Visualizations")
    if model is None or movies_df is None:
        st.error("Model or data not available."); st.stop()

    import plotly.graph_objects as go

    tab1, tab2, tab3 = st.tabs(["🔥 Similarity Heatmap","🌐 Genre Co-occurrence","📉 User–Movie Matrix"])

    with tab1:
        st.markdown("### Content-Based Cosine Similarity Heatmap")
        n     = st.slider("Movies to display", 10, 50, 20)
        cb    = model["cb_matrix"]
        cbids = model["cb_movie_ids"]
        ids   = cbids[:n]
        mat   = cb[:n, :n]
        idt   = dict(zip(movies_df["movie_id"], movies_df["title"]))
        lbls  = [idt.get(m, str(m))[:25] for m in ids]
        fig   = go.Figure(go.Heatmap(z=mat, x=lbls, y=lbls,
                    colorscale="Blues", zmin=0, zmax=1,
                    hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>Sim: %{z:.3f}<extra></extra>"))
        fig.update_layout(height=600, xaxis_tickangle=-45,
                          title="Content-Based Cosine Similarity")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Genre Co-occurrence Heatmap")
        gl    = GENRES[1:]
        comat = np.zeros((len(gl), len(gl)))
        for g in movies_df["genres"]:
            ps = [p for p in str(g).split("|") if p in gl]
            for i, a in enumerate(ps):
                for b in ps[i:]:
                    ia,ib = gl.index(a), gl.index(b)
                    comat[ia][ib] += 1
                    if ia != ib: comat[ib][ia] += 1
        fig2 = go.Figure(go.Heatmap(z=comat, x=gl, y=gl, colorscale="Teal",
                    hovertemplate="<b>%{y}</b>+<b>%{x}</b>: %{z:.0f}<extra></extra>"))
        fig2.update_layout(height=550, xaxis_tickangle=-45,
                           title="Genre Co-occurrence Matrix")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### User–Movie Ratings Sparsity (sample)")
        if ratings_df is not None:
            su = sorted(ratings_df["user_id"].unique())[:40]
            sm = sorted(ratings_df["movie_id"].unique())[:60]
            pv = (ratings_df[ratings_df["user_id"].isin(su) &
                             ratings_df["movie_id"].isin(sm)]
                  .pivot(index="user_id", columns="movie_id", values="rating")
                  .fillna(0))
            mt = dict(zip(movies_df["movie_id"], movies_df["title"]))
            pv.columns = [mt.get(c, str(c))[:18] for c in pv.columns]
            fig3 = go.Figure(go.Heatmap(
                z=pv.values, x=list(pv.columns),
                y=[f"User {u}" for u in pv.index],
                colorscale="YlOrRd", zmin=0, zmax=5,
                hovertemplate="<b>%{y}</b>→<b>%{x}</b> Rating:%{z}<extra></extra>"))
            fig3.update_layout(height=600, xaxis_tickangle=-45,
                               title="User × Movie Ratings (sample)")
            st.plotly_chart(fig3, use_container_width=True)
            sp = 1-(len(ratings_df)/(ratings_df["user_id"].nunique()*ratings_df["movie_id"].nunique()))
            st.metric("Matrix Sparsity", f"{sp:.1%}")
