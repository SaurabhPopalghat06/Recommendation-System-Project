import streamlit as st
import pandas as pd
import numpy as np
import pickle
import subprocess
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-SETUP
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH  = Path("data/ml-100k")
MODEL_PATH = Path("model/hybrid_model.pkl")

def run_setup():
    if not (DATA_PATH / "u.data").exists():
        with st.spinner("⬇️ First time setup: downloading dataset (~5 MB)..."):
            r = subprocess.run([sys.executable, "download_data.py"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                st.error(f"Download failed:\n{r.stderr}"); st.stop()
    if not MODEL_PATH.exists():
        with st.spinner("🧠 Training model for the first time (~30 sec)..."):
            r = subprocess.run([sys.executable, "train.py"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                st.error(f"Training failed:\n{r.stderr}"); st.stop()
        st.success("✅ Ready!"); st.rerun()

run_setup()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="CineMatch", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* ── global ── */
    [data-testid="stAppViewContainer"] { background: #f5f6fa; }
    [data-testid="stSidebar"] { background: #0f3460 !important; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 1rem; padding: 6px 0; }
    [data-testid="stSidebar"] hr { border-color: #ffffff30; }

    /* ── header ── */
    .app-header {
        background: linear-gradient(135deg, #0f3460, #16213e);
        color: white; padding: 1.6rem 2rem; border-radius: 14px;
        margin-bottom: 1.8rem;
    }
    .app-header h1 { margin: 0; font-size: 2rem; }
    .app-header p  { margin: 4px 0 0; opacity: .75; font-size: .95rem; }

    /* ── movie list item ── */
    .movie-item {
        background: white; border-radius: 10px; padding: .9rem 1.1rem;
        margin-bottom: .5rem; border: 1px solid #e8e8e8;
        display: flex; align-items: center; justify-content: space-between;
    }
    .movie-item-title { font-weight: 600; font-size: .95rem; color: #1a1a2e; }
    .movie-item-genre { font-size: .8rem; color: #888; margin-top: 2px; }

    /* ── rec card ── */
    .rec-card {
        background: white; border-radius: 12px; padding: 1.1rem 1.3rem;
        margin-bottom: .7rem; border: 1px solid #e8e8e8;
        border-left: 4px solid #0f3460;
    }
    .rec-title { font-size: 1rem; font-weight: 700; color: #1a1a2e; }
    .rec-genre { font-size: .8rem; color: #888; margin: 3px 0 8px; }
    .pct-bar-bg {
        background: #eef0f7; border-radius: 20px; height: 8px; overflow: hidden;
    }
    .pct-bar-fill {
        height: 8px; border-radius: 20px;
        background: linear-gradient(90deg, #0f3460, #4a90d9);
    }
    .pct-label {
        font-size: .85rem; font-weight: 700; color: #0f3460; margin-bottom: 4px;
    }

    /* ── info box ── */
    .info-box {
        background: #e8f0fe; border-radius: 10px; padding: .9rem 1.2rem;
        border-left: 4px solid #4a90d9; font-size: .9rem; color: #1a1a2e;
        margin-bottom: 1rem;
    }
    .warn-box {
        background: #fff8e1; border-radius: 10px; padding: .9rem 1.2rem;
        border-left: 4px solid #f9a825; font-size: .9rem; color: #5d4037;
        margin-bottom: 1rem;
    }

    /* ── stat pill ── */
    .stat-pill {
        display: inline-block; background: #0f3460; color: white;
        border-radius: 20px; padding: 3px 14px; font-size: .8rem;
        font-weight: 600; margin: 3px 3px 0 0;
    }

    /* ── buttons ── */
    .stButton > button {
        border-radius: 8px; font-weight: 600; font-size: .95rem;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label { font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists(): return None
    with open(MODEL_PATH, "rb") as f: return pickle.load(f)

@st.cache_data
def load_movies():
    if not (DATA_PATH / "u.item").exists(): return None, []
    mcols  = ["movie_id","title","release_date","video_date","url"] + [f"g{i}" for i in range(19)]
    movies = pd.read_csv(DATA_PATH/"u.item", sep="|", names=mcols, encoding="latin-1")
    GENRES = ["unknown","Action","Adventure","Animation","Children's","Comedy",
              "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
              "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]
    movies["genres"] = movies[[f"g{i}" for i in range(19)]].apply(
        lambda r: ", ".join([GENRES[i] for i, v in enumerate(r) if v == 1]), axis=1
    )
    movies["year"] = movies["title"].str.extract(r'\((\d{4})\)').fillna("").astype(str)
    movies["display"] = movies["title"]
    return movies, GENRES[1:]

# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def get_recommendations(model, user_ratings: dict, movies_df, top_n=10):
    """Returns top_n recommendations with match % for each."""
    if model is None or len(user_ratings) < 5:
        return pd.DataFrame()

    U          = model["U"]
    Vt         = model["Vt"]
    movie_ids  = model["movie_ids"]
    user_means = model["user_means"]
    cb_matrix  = model["cb_matrix"]
    cb_ids     = model["cb_movie_ids"]
    cf_weight  = model.get("cf_weight", 0.6)
    cb_weight  = 1.0 - cf_weight

    pseudo_mean = float(np.mean(list(user_ratings.values())))
    mid_to_cf   = {m: i for i, m in enumerate(movie_ids)}
    cb_mid_to_i = {m: i for i, m in enumerate(cb_ids)}

    # Build pseudo-user vector & project into latent space
    user_vec = np.zeros(len(movie_ids))
    for mid, r in user_ratings.items():
        if mid in mid_to_cf:
            user_vec[mid_to_cf[mid]] = r - pseudo_mean

    cf_scores_all = (user_vec @ Vt.T) @ Vt
    rated_set     = set(user_ratings.keys())
    scores        = {}

    for mid in movie_ids:
        if mid in rated_set:
            continue
        ci       = mid_to_cf[mid]
        cf_score = float(np.clip(cf_scores_all[ci] + pseudo_mean, 1.0, 5.0))

        cb_score = 0.0
        if mid in cb_mid_to_i:
            cidx     = cb_mid_to_i[mid]
            sim_vals = []
            for rated_id, rating in user_ratings.items():
                if rated_id in cb_mid_to_i:
                    ridx = cb_mid_to_i[rated_id]
                    sim  = float(cb_matrix[cidx, ridx])
                    sim_vals.append(sim * (rating / 5.0))
            if sim_vals:
                cb_score = float(np.mean(sim_vals)) * 5.0

        hybrid = cf_weight * cf_score + cb_weight * cb_score
        scores[mid] = hybrid

    if not scores:
        return pd.DataFrame()

    top_ids    = sorted(scores, key=scores.get, reverse=True)[:top_n]
    max_score  = max(scores.values()) if scores else 5.0
    min_score  = min(scores.values()) if scores else 1.0
    score_range = max_score - min_score if max_score != min_score else 1.0

    rows = []
    for mid in top_ids:
        raw  = scores[mid]
        pct  = int(((raw - min_score) / score_range) * 55 + 45)   # scale to 45–100%
        pct  = min(pct, 99)
        rows.append({"movie_id": mid, "match_pct": pct, "raw": round(raw, 3)})

    result = pd.DataFrame(rows)
    result = result.merge(
        movies_df[["movie_id","title","genres","year"]], on="movie_id", how="left"
    )
    return result

# ══════════════════════════════════════════════════════════════════════════════
# INIT SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "my_list"  not in st.session_state: st.session_state.my_list  = {}   # {movie_id: rating}
if "show_recs" not in st.session_state: st.session_state.show_recs = False

model            = load_model()
movies_df, GENRES = load_movies()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.markdown("---")
    page = st.radio("", ["🏠 Home", "📋 My Movie List", "✨ Discover Movies"],
                    label_visibility="collapsed")
    st.markdown("---")
    n = len(st.session_state.my_list)
    st.markdown(f"**Movies in your list:** {n}")
    if n >= 5:
        st.markdown("✅ Ready to get recommendations!")
    else:
        st.markdown(f"⚠️ Add {5-n} more movie(s) to unlock recommendations.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="app-header">
      <h1>🎬 CineMatch</h1>
      <p>Smart movie recommendations powered by Hybrid AI</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        ### 📋 Step 1 — Build your list
        Go to **My Movie List**, search for movies you've enjoyed and rate them (1–5 stars).
        Add **at least 5 movies** to unlock recommendations.
        """)
    with c2:
        st.markdown("""
        ### ✨ Step 2 — Discover
        Hit **"Suggest Movies I'll Love"** and the AI analyses your taste — 
        combining what similar users liked with movies that match your genre preferences.
        """)
    with c3:
        st.markdown("""
        ### 🔄 Step 3 — Refine anytime
        Add more movies, change your ratings, or clear the list and start fresh.
        Your recommendations update instantly every time.
        """)

    st.markdown("---")

    st.markdown("### 🧠 How the AI works")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Collaborative Filtering (SVD)**  
        Learns hidden patterns from 100,000 ratings by 943 users.
        Finds what people with similar taste to you enjoyed.
        """)
    with c2:
        st.markdown("""
        **Content-Based Filtering (TF-IDF)**  
        Analyses the genres of movies you liked and finds
        other movies with a matching genre profile.
        """)

    st.markdown("""
    <div class="info-box">
    💡 The final score is a <b>60% CF + 40% Content</b> blend — giving you the best of both worlds.
    The <b>match %</b> shown on each recommendation tells you how well it fits your taste profile.
    </div>
    """, unsafe_allow_html=True)

    if movies_df is not None:
        st.markdown("---")
        st.markdown("### 📊 Dataset at a glance")
        ratings = pd.read_csv(DATA_PATH/"u.data", sep="\t",
                              names=["user_id","movie_id","rating","timestamp"])
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🎬 Movies",      f"{movies_df['movie_id'].nunique():,}")
        c2.metric("👥 Users",       f"{ratings['user_id'].nunique():,}")
        c3.metric("⭐ Total Ratings",f"{len(ratings):,}")
        c4.metric("📊 Avg Rating",  f"{ratings['rating'].mean():.2f} / 5")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MY MOVIE LIST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 My Movie List":
    st.markdown("""
    <div class="app-header">
      <h1>📋 My Movie List</h1>
      <p>Add movies you've watched and rate them — the more you add, the better your recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    if movies_df is None:
        st.error("Movie data not available."); st.stop()

    # ── Add a movie ───────────────────────────────────────────────────────────
    st.markdown("### ➕ Add a Movie")

    col_search, col_genre = st.columns([3, 2])
    with col_search:
        search_q = st.text_input("Search by title", placeholder="e.g. Toy Story, Batman...",
                                 label_visibility="collapsed")
    with col_genre:
        genre_f = st.selectbox("Filter by genre", ["All genres"] + GENRES,
                               label_visibility="collapsed")

    # Filter movie pool
    pool = movies_df.copy()
    if search_q:
        pool = pool[pool["title"].str.contains(search_q, case=False, na=False)]
    if genre_f != "All genres":
        pool = pool[pool["genres"].str.contains(genre_f, case=False, na=False)]

    already_added = set(st.session_state.my_list.keys())
    pool = pool[~pool["movie_id"].isin(already_added)]

    col_pick, col_star, col_add = st.columns([4, 2, 1])
    with col_pick:
        options = ["— select a movie —"] + pool["title"].head(100).tolist()
        chosen  = st.selectbox("Movie", options, label_visibility="collapsed")
    with col_star:
        stars = st.select_slider("Rating", options=[1,2,3,4,5], value=4,
                                  label_visibility="collapsed",
                                  format_func=lambda x: "★"*x + "☆"*(5-x))
    with col_add:
        add_clicked = st.button("Add", type="primary", use_container_width=True)

    if add_clicked and chosen != "— select a movie —":
        row = movies_df[movies_df["title"] == chosen].iloc[0]
        st.session_state.my_list[int(row["movie_id"])] = stars
        st.session_state.show_recs = False
        st.success(f"✅ Added **{chosen}** with {'★'*stars}")
        st.rerun()

    st.markdown("---")

    # ── Current list ──────────────────────────────────────────────────────────
    my_list = st.session_state.my_list
    n = len(my_list)

    col_title, col_action = st.columns([3,1])
    with col_title:
        st.markdown(f"### 🎞️ Your List &nbsp; <span class='stat-pill'>{n} movies</span>", unsafe_allow_html=True)
    with col_action:
        if n > 0:
            if st.button("🗑️ Clear entire list", use_container_width=True):
                st.session_state.my_list = {}
                st.session_state.show_recs = False
                st.rerun()

    if n == 0:
        st.markdown("""
        <div class="warn-box">
        Your list is empty. Search for movies above and start adding them!
        </div>""", unsafe_allow_html=True)
    else:
        # progress toward 5
        if n < 5:
            st.markdown(f"""
            <div class="warn-box">
            ⚠️ You need at least <b>5 movies</b> to get recommendations.
            Add <b>{5 - n} more</b> to unlock the Discover page.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-box">
            ✅ Great! You have {n} movies. Head to <b>✨ Discover Movies</b> to get your recommendations.
            </div>""", unsafe_allow_html=True)

        # List items with inline rating edit + remove
        mid_to_title = dict(zip(movies_df["movie_id"], movies_df["title"]))
        mid_to_genre = dict(zip(movies_df["movie_id"], movies_df["genres"]))

        to_remove = None
        for mid, rating in list(my_list.items()):
            title = mid_to_title.get(mid, f"Movie {mid}")
            genre = mid_to_genre.get(mid, "")[:45]
            c1, c2, c3 = st.columns([5, 2, 1])
            with c1:
                st.markdown(f"""
                <div style="padding:6px 0">
                  <div class="movie-item-title">{title}</div>
                  <div class="movie-item-genre">{genre}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                new_r = st.select_slider("", options=[1,2,3,4,5], value=rating,
                                          key=f"edit_{mid}", label_visibility="collapsed",
                                          format_func=lambda x: "★"*x)
                if new_r != rating:
                    st.session_state.my_list[mid] = new_r
                    st.session_state.show_recs = False
                    st.rerun()
            with c3:
                if st.button("✕", key=f"rm_{mid}", use_container_width=True):
                    to_remove = mid

        if to_remove is not None:
            del st.session_state.my_list[to_remove]
            st.session_state.show_recs = False
            st.rerun()

    # ── Suggest button ────────────────────────────────────────────────────────
    st.markdown("---")
    if n >= 5:
        if st.button("✨ Suggest Movies Based on My List", type="primary",
                     use_container_width=True):
            st.session_state.show_recs = True
            # Switch to discover page
            st.switch_page = "✨ Discover Movies"
            st.info("👉 Head to **✨ Discover Movies** in the sidebar to see your recommendations!")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DISCOVER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✨ Discover Movies":
    st.markdown("""
    <div class="app-header">
      <h1>✨ Discover Movies</h1>
      <p>Movies the AI thinks you'll love — based on your list and ratings</p>
    </div>
    """, unsafe_allow_html=True)

    my_list = st.session_state.my_list
    n       = len(my_list)

    if n == 0:
        st.markdown("""
        <div class="warn-box">
        📋 Your movie list is empty. Go to <b>My Movie List</b> and add some movies first!
        </div>""", unsafe_allow_html=True)
        st.stop()

    if n < 5:
        st.markdown(f"""
        <div class="warn-box">
        ⚠️ You only have <b>{n} movie(s)</b> in your list.
        Please add <b>{5 - n} more</b> in <b>My Movie List</b> to unlock recommendations.
        </div>""", unsafe_allow_html=True)
        st.stop()

    if model is None:
        st.error("Model not available."); st.stop()

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2 = st.columns([2, 2])
    with c1:
        top_n = st.slider("How many movies to suggest?", 5, 20, 10)
    with c2:
        cf_w = st.slider("Preference: Similar users ← → Genre match",
                         0.0, 1.0, 0.6, 0.1,
                         help="Left = more weight on users like you · Right = more weight on genre similarity")

    model["cf_weight"] = cf_w

    # ── Generate button ───────────────────────────────────────────────────────
    if st.button("🎬 Suggest Movies I'll Love", type="primary", use_container_width=True):
        st.session_state.show_recs = True

    if not st.session_state.show_recs:
        st.markdown("""
        <div class="info-box">
        👆 Click the button above to generate your personalised movie recommendations.
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── Compute & show ────────────────────────────────────────────────────────
    with st.spinner("🔍 Analysing your taste profile..."):
        recs = get_recommendations(model, my_list, movies_df, top_n)

    if recs.empty:
        st.warning("No recommendations found. Try adding more diverse movies to your list.")
        st.stop()

    # Your list summary
    mid_to_title = dict(zip(movies_df["movie_id"], movies_df["title"]))
    liked = sorted(my_list.items(), key=lambda x: x[1], reverse=True)
    pills = " ".join([
        f"<span class='stat-pill'>{'★'*r} {mid_to_title.get(m,'?')[:22]}</span>"
        for m, r in liked
    ])
    st.markdown(f"**Based on your {n} movies:** {pills}", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"### 🎯 Top {len(recs)} Recommendations for You")

    for rank, (_, row) in enumerate(recs.iterrows(), 1):
        pct   = int(row["match_pct"])
        title = row["title"]
        genre = str(row.get("genres",""))[:50]
        year  = str(row.get("year",""))

        # colour gradient: green for high, amber for mid
        bar_color = ("#1a8a4a" if pct >= 80
                     else "#4a90d9" if pct >= 65
                     else "#e07b00")

        st.markdown(f"""
        <div class="rec-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
              <div class="rec-title">#{rank} &nbsp; {title}
                <span style="font-weight:400;font-size:.82rem;color:#999"> {year}</span>
              </div>
              <div class="rec-genre">{genre}</div>
            </div>
            <div style="text-align:right;min-width:60px">
              <div style="font-size:1.4rem;font-weight:800;color:{bar_color}">{pct}%</div>
              <div style="font-size:.72rem;color:#aaa">match</div>
            </div>
          </div>
          <div class="pct-bar-bg">
            <div class="pct-bar-fill" style="width:{pct}%;background:{bar_color}"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("💡 Tip: Update your list anytime in **My Movie List** — your recommendations will refresh automatically.")
