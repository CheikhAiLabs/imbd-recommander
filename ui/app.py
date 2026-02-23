"""
IMDb Recommender UI
====================
Modern, cinema-grade Streamlit interface.
Dark theme with IMDb-gold accent, fuzzy search, and polished movie cards.
"""

import os
import time
from typing import Optional

import requests
import streamlit as st
from streamlit_searchbox import st_searchbox

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IMDb Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:9876")

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    /* ── Header ──────────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
    }
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f5c518 0%, #ffdb4d 40%, #e8a317 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1.1rem;
        color: #8899a6;
        font-weight: 300;
    }

    /* ── Movie Card ──────────────────────────────────────── */
    .movie-card {
        background: linear-gradient(145deg, #1a2332 0%, #111a24 100%);
        border: 1px solid #2a3a4a;
        border-radius: 16px;
        padding: 1.2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        margin-bottom: 1rem;
        height: 100%;
    }
    .movie-card:hover {
        border-color: #f5c518;
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(245, 197, 24, 0.15);
    }

    .mc-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.7rem;
    }
    .mc-rank {
        background: rgba(0, 0, 0, 0.5);
        color: #f5c518;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 800;
        border: 2px solid #f5c518;
        flex-shrink: 0;
    }
    .mc-score {
        background: linear-gradient(135deg, #f5c518, #e8a317);
        color: #000;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    .mc-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 0.6rem;
        line-height: 1.35;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    .mc-meta {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin-bottom: 0.6rem;
    }
    .mc-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.2rem;
        padding: 0.15rem 0.5rem;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .mc-year {
        background: rgba(245, 197, 24, 0.15);
        color: #f5c518;
    }
    .mc-rating {
        background: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
    }
    .mc-votes {
        background: rgba(52, 152, 219, 0.15);
        color: #3498db;
    }
    .mc-runtime {
        background: rgba(155, 89, 182, 0.15);
        color: #9b59b6;
    }

    .mc-genres {
        display: flex;
        gap: 0.3rem;
        flex-wrap: wrap;
    }
    .mc-genre-tag {
        padding: 0.15rem 0.5rem;
        border-radius: 8px;
        font-size: 0.68rem;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.06);
        color: #99aabb;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* ── IMDb Link Icon ──────────────────────────────────── */
    .mc-imdb-link {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.7rem;
        color: #f5c518;
        text-decoration: none;
        opacity: 0.7;
        transition: opacity 0.2s;
    }
    .mc-imdb-link:hover { opacity: 1; }

    /* ── Query Hero Card ─────────────────────────────────── */
    .query-hero {
        background: linear-gradient(145deg, #1a2636 0%, #0f1923 100%);
        border: 2px solid #f5c518;
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    .query-hero h2 {
        color: #f5c518;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    .query-hero .qh-meta {
        color: #8899a6;
        font-size: 0.95rem;
    }

    /* ── Stats Bar ───────────────────────────────────────── */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 2.5rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stat-item { text-align: center; }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f5c518;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #8899a6;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Section Title ───────────────────────────────────── */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #fff;
        margin: 1.5rem 0 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(245,197,24,0.4), transparent);
    }

    /* ── Footer ──────────────────────────────────────────── */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #556677;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 3rem;
    }

    /* ── Error Card ──────────────────────────────────────── */
    .error-card {
        background: rgba(231, 76, 60, 0.1);
        border: 1px solid rgba(231, 76, 60, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #e74c3c;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    .sidebar-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1rem;
    }

    /* Streamlit overrides */
    .stSelectbox label, .stSlider label { font-weight: 500 !important; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Helper Functions ──────────────────────────────────────────────────────────


def check_api_health() -> Optional[dict]:
    """Check if the API is running and healthy."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.ConnectionError:
        return None
    return None


def search_titles(query: str, limit: int = 20) -> list[dict]:
    """Search for titles via the API (supports fuzzy/typo-tolerant search)."""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/search",
            params={"q": query, "limit": limit},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.ConnectionError:
        pass
    return []


def get_recommendations(title: str, top_k: int = 10, tconst: str = "") -> Optional[dict]:
    """Get recommendations from the API."""
    try:
        payload = {"title": title, "top_k": top_k}
        if tconst:
            payload["tconst"] = tconst
        resp = requests.post(
            f"{API_BASE_URL}/recommend",
            json=payload,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            st.warning("No results found for this title. Please try another search.")
        else:
            st.error("Something went wrong. Please try again.")
    except requests.ConnectionError:
        st.error("Service temporarily unavailable. Please try again later.")
    return None


def format_votes(votes: int) -> str:
    """Format vote count for display."""
    if votes >= 1_000_000:
        return f"{votes / 1_000_000:.1f}M"
    elif votes >= 1_000:
        return f"{votes / 1_000:.0f}K"
    return str(votes)


def render_movie_card(rec: dict, show_rank: bool = True, show_score: bool = True):
    """Render a polished movie card."""
    title = rec.get("title", "Unknown")
    year = rec.get("year")
    rating = rec.get("rating")
    votes = rec.get("votes")
    genres = rec.get("genres", "")
    runtime = rec.get("runtime_minutes")
    score = rec.get("similarity_score")
    rank = rec.get("rank")
    tconst = rec.get("tconst", "")

    # Header: rank + score
    rank_html = f'<div class="mc-rank">#{rank}</div>' if show_rank and rank else "<div></div>"
    score_html = ""
    if show_score and score is not None:
        pct = int(score * 100)
        score_html = f'<div class="mc-score">{pct}% match</div>'

    # Meta badges
    meta_parts = []
    if year:
        meta_parts.append(f'<span class="mc-badge mc-year">📅 {year}</span>')
    if rating:
        meta_parts.append(f'<span class="mc-badge mc-rating">⭐ {rating:.1f}</span>')
    if votes:
        meta_parts.append(f'<span class="mc-badge mc-votes">🗳️ {format_votes(votes)}</span>')
    if runtime:
        meta_parts.append(f'<span class="mc-badge mc-runtime">⏱ {runtime}m</span>')
    meta_html = f'<div class="mc-meta">{"".join(meta_parts)}</div>' if meta_parts else ""

    # Genre tags
    genre_html = ""
    if genres:
        genre_list = genres.split(",") if isinstance(genres, str) else genres
        tags = "".join(f'<span class="mc-genre-tag">{g.strip()}</span>' for g in genre_list[:5])
        genre_html = f'<div class="mc-genres">{tags}</div>'

    # IMDb link
    imdb_html = ""
    if tconst:
        imdb_html = f'<a class="mc-imdb-link" href="https://www.imdb.com/title/{tconst}/" target="_blank">🔗 View on IMDb</a>'

    card_html = f"""
    <div class="movie-card">
        <div class="mc-header">{rank_html}{score_html}</div>
        <div class="mc-title">{title}</div>
        {meta_html}
        {genre_html}
        {imdb_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# ─── Main App ──────────────────────────────────────────────────────────────────


def main():
    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🎬 IMDb Recommender</h1>
            <p>Discover your next favorite movie or series with AI-powered recommendations</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check API health
    health = check_api_health()

    if health is None:
        st.markdown(
            """
            <div class="error-card">
                <h3>⚠️ Service Unavailable</h3>
                <p>The recommendation service is temporarily unavailable. Please try again later.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-header"><h2 style="color: #f5c518; margin: 0;">⚙️ Settings</h2></div>',
            unsafe_allow_html=True,
        )

        top_k = st.slider(
            "Number of Recommendations",
            min_value=3,
            max_value=30,
            value=10,
            step=1,
            help="How many similar titles to show",
        )

        cols_count = st.radio(
            "Grid Layout",
            options=[3, 4, 5],
            index=1,
            horizontal=True,
            help="Number of cards per row",
        )

        st.divider()

        st.markdown("### 📊 System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Status", "🟢 Online")
        with col2:
            st.metric("Titles", f"{health.get('titles_count', 0):,}")

        st.divider()

        st.markdown(
            """
            ### 🎯 How it works
            1. **Search** for a movie or series
            2. **Select** a title from the suggestions
            3. **Discover** similar titles you'll love
            """,
        )

        st.divider()
        st.markdown(
            '<div class="footer">Built with ❤️ using FastAPI & Streamlit</div>',
            unsafe_allow_html=True,
        )

    # ─── Search Section ────────────────────────────────────────────
    st.markdown("### 🔍 Find a Movie or Series")

    def search_autocomplete(query: str) -> list[tuple[str, dict]]:
        """Called on every keystroke — returns (label, value) for the dropdown."""
        if not query or len(query) < 2:
            return []
        results = search_titles(query, limit=15)
        options = []
        for r in results:
            label = r["title"]
            if r.get("year"):
                label += f" ({r['year']})"
            if r.get("rating"):
                label += f" ⭐ {r['rating']}"
            if r.get("genres"):
                label += f" — {r['genres']}"
            options.append((label, r))
        return options

    selected_title = st_searchbox(
        search_autocomplete,
        placeholder="Search a movie or series (e.g., Inception, Breaking Bad, Parasite)...",
        label="Search for a title",
        clear_on_submit=False,
        key="movie_searchbox",
    )

    # ─── Recommendations ──────────────────────────────────────────
    if selected_title and isinstance(selected_title, dict):
        st.divider()

        title_name = selected_title["title"]

        if st.button(
            f'🎯 Get Recommendations for "{title_name}"',
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Finding similar titles..."):
                result = get_recommendations(
                    title_name,
                    top_k=top_k,
                    tconst=selected_title.get("tconst", ""),
                )

            if result:
                query_info = result.get("query", {})
                latency = result.get("latency_ms", 0)
                recs = result["recommendations"]

                # Stats bar
                top_score = recs[0]["similarity_score"] if recs else 0
                st.markdown(
                    f"""
                    <div class="stats-bar">
                        <div class="stat-item">
                            <div class="stat-value">{len(recs)}</div>
                            <div class="stat-label">Results</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{latency:.0f}ms</div>
                            <div class="stat-label">Latency</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{int(top_score * 100)}%</div>
                            <div class="stat-label">Best Match</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{query_info.get('rating', 'N/A')}</div>
                            <div class="stat-label">Rating</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Query hero card
                q_genres = query_info.get("genres", "")
                q_year = query_info.get("year", "")
                q_rating = query_info.get("rating", "")
                q_votes = query_info.get("votes")
                q_runtime = query_info.get("runtime_minutes")
                q_tconst = query_info.get("tconst", "")

                meta_parts = []
                if q_year:
                    meta_parts.append(str(q_year))
                if q_genres:
                    meta_parts.append(q_genres)
                if q_rating:
                    meta_parts.append(f"⭐ {q_rating}")
                if q_votes:
                    meta_parts.append(f"🗳️ {format_votes(q_votes)} votes")
                if q_runtime:
                    meta_parts.append(f"⏱ {q_runtime}min")

                imdb_link = ""
                if q_tconst:
                    imdb_link = f' • <a href="https://www.imdb.com/title/{q_tconst}/" target="_blank" style="color: #f5c518;">View on IMDb ↗</a>'

                st.markdown(
                    f"""
                    <div class="query-hero">
                        <h2>🎬 {query_info.get('title', title_name)}</h2>
                        <div class="qh-meta">{' • '.join(meta_parts)}{imdb_link}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Section title
                st.markdown(
                    '<div class="section-title">🏆 Top Recommendations</div>',
                    unsafe_allow_html=True,
                )

                # Movie grid
                cols = st.columns(cols_count)
                for i, rec in enumerate(recs):
                    with cols[i % cols_count]:
                        render_movie_card(rec)

    # Footer
    st.markdown(
        """
        <div class="footer">
            <p>
                Data sourced from
                <a href="https://www.imdb.com" style="color: #f5c518;" target="_blank">IMDb</a>
                Non-Commercial Datasets
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
