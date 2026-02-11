import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Face Value – Emotion in Movies",
    layout="wide",
)

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

EMOTION_COLORS = {
    "angry": "#E74C3C",
    "fear": "#7F8C8D",
    "happy": "#FFD93D",
    "sad": "#4A90E2",
    "surprise": "#9B59B6",
}

EMOTIONS = ["angry", "fear", "happy", "sad", "surprise"]
MOVIE_GROUPS = {
    "Featured Selection": [
        "finding_nemo", "airplane", "inside_out", "real_steel",
        "dark_knight", "pulp_fiction", "dodgeball", "inception"
    ],
    "Comedies": [
        "airplane", "dodgeball", "old_school", "pineapple_exp",
        "boondock_saints", "hot_fuzz", "seven_psychopaths", "scott_pilgrim"
    ],
    "Kids Movies": [
        "finding_nemo", "big_hero_6", "inside_out", "lego_movie",
        "frankenweenie"
    ],
    "Harry Potter Series": [
        "hp1_sorcerers_stone", "hp2_chamber_of_secrets",
        "hp3_prisoner_of_azkaban", "hp4_goblet_of_fire",
        "hp5_order_phoenix", "hp6_half_blood_prince",
        "hp7_deathly_hallows_part_1", "hp7_deathly_hallows_part_2"
    ],
    "Dark Dramas": [
        "dark_knight", "dark_knight_rises", "inception",
        "black_mass", "pulp_fiction", "lucky_number_slevin"
    ],
    "Lord of the Rings": ["lotr_1", "lotr_2", "lotr_3"],
}

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data(file_name):
    base = Path("movie_data")
    fv = pd.read_csv(base / "FaceValue" / file_name)
    raf = pd.read_csv(base / "RAFDB" / file_name)
    return fv, raf

fv_time, raf_time = load_data("all_movies.csv")
fv_tidy, raf_tidy = load_data("movie_emotion_summary.csv")

ALL_MOVIES = sorted(fv_tidy["movie"].unique())

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def emotion_legend():
    cols = st.columns(len(EMOTIONS))
    for i, e in enumerate(EMOTIONS):
        with cols[i]:
            st.markdown(
                f"""
                <div style="
                    background:{EMOTION_COLORS[e]};
                    padding:10px;
                    border-radius:6px;
                    text-align:center;
                    font-weight:600;
                    color:white;
                ">
                    {e.title()}
                </div>
                """,
                unsafe_allow_html=True,
            )

def prepare_stacked_data(df, movies):
    df = df[df["movie"].isin(movies)]
    tidy = pd.melt(
        df,
        id_vars="movie",
        value_vars=[f"{e}_pct" for e in EMOTIONS],
        var_name="emotion",
        value_name="percent",
    )
    tidy["emotion"] = tidy["emotion"].str.replace("_pct", "")
    tidy.sort_values(by="movie", inplace=True)
    return tidy

def stacked_bar(tidy, title):
    fig = px.bar(
        tidy,
        x="movie",
        y="percent",
        color="emotion",
        barmode="stack",
        color_discrete_map=EMOTION_COLORS,
        category_orders={"emotion": EMOTIONS},
        height=550,
        title=title,
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(showlegend=False)
    return fig

def prepare_timeline(df, movie, threshold):
    df = df[(df["movie"] == movie) & (df["confidence"] >= threshold)].copy()
    df.sort_values("timestamp_sec", inplace=True)

    if df.empty:
        return None

    for e in EMOTIONS:
        df[f"{e}_count"] = (df["emotion"] == e).cumsum()

    df["timestamp_min"] = df["timestamp_sec"] / 60
    return df

def timeline_plot(df, movie, threshold):
    fig = go.Figure()
    for e in EMOTIONS:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_min"],
                y=df[f"{e}_count"],
                mode="lines+text",
                line=dict(color=EMOTION_COLORS[e], width=2.5),
                name=e
            )
        )

    fig.update_layout(
        title=f"{movie.replace('_',' ').title()} (confidence ≥ {threshold})",
        xaxis_title="Time (minutes)",
        yaxis_title="Total faces detected",
        hovermode="x unified",
        height=500,
        showlegend=False,
    )
    return fig

# =============================================================================
# SESSION STATE
# =============================================================================

def init_state():
    defaults = {
        "compare_group": "Comedies",
        "compare_movies": MOVIE_GROUPS["Comedies"][:6],
        "compare_model": "Side-by-side",
        "timeline_movie": ALL_MOVIES[0],
        "timeline_model": "Face Value",
        "timeline_conf": 0.5,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()

# =============================================================================
# HEADER
# =============================================================================

st.title("Face Value: Emotion in Movies")
st.markdown(
    """
    This project explores how different emotion recognition models behave
    when applied to real movies.  
    Rather than focusing on test accuracy alone, it asks a simpler question:

    **Do model predictions make sense in context?**
    """
)

emotion_legend()
st.markdown("---")

# =============================================================================
# TABS
# =============================================================================

tab_overview, tab_explore, tab_timeline, tab_insight, tab_methods = st.tabs(
    [
        "🎬 Overview",
        "📊 Movie Explorer",
        "📈 Timeline Stories",
        "🔍 Model Insight",
        "🧭 Methods & Limits",
    ]
)

# =============================================================================
# TAB 1 — OVERVIEW (STATIC)
# =============================================================================

with tab_overview:
    st.subheader("What does emotion look like across movies?")

    st.markdown(
        """
        Below is a comparison of emotion predictions across a small set of
        well-known **comedy films**.

        Both models analyze the *same movie frames*.  
        The difference is how they were trained.
        """
    )

    st.markdown("#### Emotion distribution by movie (curated example)")

    fv_comedies = Image.open("images/fv_comedies.png")
    raf_comedies = Image.open("images/raf_comedies.png")
    l_img, r_img = st.columns(2)
    with l_img:
        st.header("Face Value")
        st.image(fv_comedies)
    with r_img:
        st.header("RAF DB")
        st.image(raf_comedies)

    # st.markdown(
    #     """
    #     *Tip:* This should show Face Value and RAF-DB side by side for the
    #     same group of movies.
    #     """
    # )

    st.markdown(
        """
        **What to notice:**
        - Comedies are expected to show a mix of happy, surprise, and lighter emotions
        - One model tends to collapse toward a single emotion
        - The other shows broader emotional variety that better matches genre expectations
        """
    )

# =============================================================================
# TAB 2 — MOVIE EXPLORER (INTERACTIVE)
# =============================================================================

with tab_explore:
    with st.expander("⚙️ Explore settings", expanded=True):
        c1, c2, c3 = st.columns([1.5, 2, 3])

        with c1:
            st.session_state.compare_model = st.radio(
                "View",
                ["Face Value", "RAF-DB", "Side-by-side"],
                horizontal=True,
                key="compare_model_radio",
            )

        with c2:
            st.session_state.compare_group = st.selectbox(
                "Movie group",
                list(MOVIE_GROUPS.keys()),
            )

        with c3:
            defaults = MOVIE_GROUPS[st.session_state.compare_group]
            st.session_state.compare_movies = st.multiselect(
                "Movies",
                ALL_MOVIES,
                default=defaults,
                max_selections=10,
                format_func=lambda x: x.replace("_", " ").title(),
            )

    st.markdown("---")

    movies = st.session_state.compare_movies
    model = st.session_state.compare_model

    if not movies:
        st.warning("Select at least one movie to continue.")
        st.stop()

    if model == "Side-by-side":
        l, r = st.columns(2)
        with l:
            st.plotly_chart(
                stacked_bar(
                    prepare_stacked_data(fv_tidy, movies),
                    "Face Value",
                ),
                width='stretch',
            )
        with r:
            st.plotly_chart(
                stacked_bar(
                    prepare_stacked_data(raf_tidy, movies),
                    "RAF-DB",
                ),
                width='stretch',
            )
    elif model == "Face Value":
        st.plotly_chart(
            stacked_bar(
                prepare_stacked_data(fv_tidy, movies),
                "Face Value",
            ),
            width='stretch',
        )
    else:
        st.plotly_chart(
            stacked_bar(
                prepare_stacked_data(raf_tidy, movies),
                "RAF-DB",
            ),
            width='stretch',
        )

# =============================================================================
# TAB 3 — TIMELINE STORIES (INTERACTIVE)
# =============================================================================

with tab_timeline:
    with st.expander("⚙️ Timeline settings", expanded=True):
        c1, c2, c3 = st.columns([1.5, 2.5, 2])

        with c1:
            st.session_state.timeline_model = st.radio(
                "View",
                ["Face Value", "RAF-DB", "Side-by-side"],
                horizontal=True,
                key="timeline_model_radio",
            )

        with c2:
            st.session_state.timeline_movie = st.selectbox(
                "Movie",
                ALL_MOVIES,
                format_func=lambda x: x.replace("_", " ").title(),
            )

        with c3:
            st.session_state.timeline_conf = st.slider(
                "Minimum confidence",
                0.0, 1.0, st.session_state.timeline_conf, 0.05
            )

    st.markdown("---")

    movie = st.session_state.timeline_movie
    conf = st.session_state.timeline_conf
    model = st.session_state.timeline_model

    def show_timeline(df):
        data = prepare_timeline(df, movie, conf)
        if data is None:
            st.warning("No data available at this confidence level.")
        else:
            st.plotly_chart(
                timeline_plot(data, movie, conf),
                width='stretch',
            )

    if model == "Side-by-side":
        l, r = st.columns(2)
        with l:
            show_timeline(fv_time)
        with r:
            show_timeline(raf_time)
    elif model == "Face Value":
        show_timeline(fv_time)
    else:
        show_timeline(raf_time)

# =============================================================================
# TAB 4 — MODEL INSIGHT (STATIC)
# =============================================================================

with tab_insight:
    st.subheader("How confident are the models?")

    st.markdown(
        """
        These figures look at *how confident* each model is when it predicts
        different emotions.
        """
    )

    fv_conf = Image.open("images/FV_pred_conf_by_emo.png")
    raf_conf = Image.open("images/RAF_pred_conf_by_emo.png")
    fv_img, raf_img = st.columns(2)
    with fv_img:
        st.header("Face Value")
        st.image(fv_conf)
    with raf_img:
        st.header("RAF DB")
        st.image(raf_conf)

    st.markdown(
        """
        **Key idea:**  
        Higher confidence does not automatically mean better real-world behavior.
        Confidence distributions reveal bias, collapse, and calibration issues.
        """
    )

    st.markdown("---")

    st.subheader("Face Value Validation")
    fv_validation = Image.open("images/fv_val_cm.png")
    st.header("Face Value Validation")
    st.image(fv_validation)


    st.markdown("---")

    st.subheader("Where do models make mistakes?")

    fv_confusion = Image.open("images/fv_raf_test_cm.png")
    raf_confusion = Image.open("images/raf_raf_test_cm.png")
    fv_cm, raf_cm = st.columns(2)
    with fv_cm:
        st.header("Face Value")
        st.image(fv_confusion)
    with raf_cm:
        st.header("RAF DB")
        st.image(raf_confusion)

# =============================================================================
# TAB 5 — METHODS & LIMITS (STATIC)
# =============================================================================

with tab_methods:
    st.subheader("How this works (and where it breaks)")

    st.markdown(
        """
        This project uses a **weak supervision** approach:
        images are collected using emotion-related keywords rather than manual labels.

        Faces are detected automatically, and models are trained using standard
        deep learning tools.
        """
    )

    st.markdown(
        """
        **Important limitations:**
        - No ground truth for movie emotions
        - Some emotions are over-represented
        - Timeline interpretation is qualitative
        - Models are not intended for production use
        """
    )

    st.markdown(
        """
        The goal is not perfect accuracy, but **meaningful patterns** in realistic settings.
        """
    )

# =============================================================================
# FOOTER
# =============================================================================

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built by <a href='https://pixelprocess.org' target='_blank'>PixelProcess</a> | 
    Part of <a href='https://dexterousdata.com' target='_blank'>Dexterous Data</a><br>
    <a href='https://github.com/pixel-process-dev/face-value' target='_blank'>View Source on GitHub</a>
</div>
""", unsafe_allow_html=True)