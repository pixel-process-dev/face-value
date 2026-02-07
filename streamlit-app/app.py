import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# ============ GLOBAL EMOTION COLOR PALETTE ============
EMOTION_COLORS = {
    "angry": "#E74C3C",      # Red - anger/aggression
    "fear": "#7F8C8D",       # Gray - uncertainty/fear
    "happy": "#FFD93D",      # Yellow - joy/happiness
    "sad": "#4A90E2",        # Blue - sadness/melancholy
    "surprise": "#9B59B6"    # Purple - unexpected/vibrant
}

# Page config
st.set_page_config(
    page_title="Face Value - Movie Emotion Analysis",
    layout="wide"
)

# Load data (cached)
@st.cache_data
def load_data(file_name):
    fv_dir = Path("movie_data/FaceValue")
    raf_dir = Path("movie_data/RAFDB")

    fv_file = fv_dir / file_name
    raf_file = raf_dir / file_name

    fv = pd.read_parquet(fv_file)
    raf = pd.read_parquet(raf_file)
    return fv, raf

fv_time, raf_time = load_data(file_name="all_movies.parquet")
fv_tidy, raf_tidy = load_data(file_name="movie_emotion_summary.parquet")

# Movie groupings
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
        "dark_knight", "dark_knight_rises", "inception", "black_mass",
        "pulp_fiction", "lucky_number_slevin"
    ],
    "Lord of the Rings": [
        "lotr_1", "lotr_2", "lotr_3"
    ],
}

def prepare_stacked_data(movie_data, movie_list):
    """Convert movie data to format for stacked bar chart"""
    # Filter to selected movies
    filtered = movie_data[movie_data['movie'].isin(movie_list)]
    perc_cols = ["angry_pct", "fear_pct", "happy_pct", "sad_pct", "surprise_pct"]
    
    tidy = pd.melt(filtered, id_vars=["movie"], value_vars=perc_cols,
        var_name="emotion", value_name="percent")
    tidy['emotion'] = tidy['emotion'].str.removesuffix('_pct')
    tidy.sort_values(by=["movie"], inplace=True)
   
    return tidy

def create_stacked_bar(tidy, title):
    """Create stacked bar chart"""
    fig = px.bar(
        tidy,
        x="movie",
        y="percent",
        color="emotion",
        barmode="stack",
        color_discrete_map=EMOTION_COLORS,  # Use global colors
        title=title,
        height=600,
        category_orders={
            "emotion": ["angry", "fear", "happy", "sad", "surprise"]
        },
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        showlegend=False,  # Remove legend
    )
    
    return fig

def prepare_timeline_data(movie_data, movie_name, confidence_threshold=0.5):
    """Prepare cumulative emotion counts for timeline plot"""
    # movie_df = (
    #     movie_data
    #     .filter(pl.col("movie") == movie_name)
    #     .filter(pl.col("confidence") >= confidence_threshold)
    #     .sort("timestamp_sec")
    # )
    movie_df = movie_data[movie_data['movie']==movie_name]
    movie_df = movie_df[movie_df['confidence']>=confidence_threshold]
    movie_df.sort_values(by="timestamp_sec", inplace=True)

    if len(movie_df) == 0:
        return None
    
    # Create cumulative counts
    emotions = ["angry", "fear", "happy", "sad", "surprise"]
    # timeline_data = []
    # cumulative = {e: 0 for e in emotions}
    
    # for row in movie_df.iter_rows(named=True):
    #     cumulative[row["emotion"]] += 1
    #     timeline_data.append({
    #         "timestamp_min": row["timestamp_sec"] / 60,
    #         **{f"{e}_count": cumulative[e] for e in emotions}
    #     })
    # Create cumulative counts for each emotion
    for emotion in emotions:
        movie_df[f"{emotion}_count"] = (movie_df["emotion"] == emotion).astype(int).cumsum()
    
    # Add timestamp in minutes
    movie_df["timestamp_min"] = movie_df["timestamp_sec"] / 60
    
    # Select only timeline columns
    timeline_df = movie_df[["timestamp_min"] + [f"{e}_count" for e in emotions]]

    return timeline_df

def create_timeline_plot(timeline_df, movie_name, confidence_threshold):
    """Create timeline plot with consistent colors"""
    if timeline_df is None:
        return None
    
    # timeline_pd = timeline_df.to_pandas()
    
    fig = go.Figure()
    
    emotions = ["angry", "fear", "happy", "sad", "surprise"]
    for emotion in emotions:
        fig.add_trace(go.Scatter(
            x=timeline_df["timestamp_min"],
            y=timeline_df[f"{emotion}_count"],
            mode="lines",
            name=emotion,
            line=dict(color=EMOTION_COLORS[emotion], width=2.5),
        ))
    
    fig.update_layout(
        title=f"Emotion Timeline: {movie_name.replace('_', ' ').title()} (confidence ≥ {confidence_threshold})",
        xaxis_title="Time (minutes)",
        yaxis_title="Cumulative Face Count",
        hovermode="x unified",
        height=500,
        showlegend=False,  # Remove legend
    )
    
    return fig

def display_emotion_legend():
    """Display color legend for emotions"""
    st.markdown("### Emotion Color Key")
    
    cols = st.columns(5)
    emotions_display = {
        "angry": "😠 Angry",
        "fear": "😨 Fear", 
        "happy": "😊 Happy",
        "sad": "😢 Sad",
        "surprise": "😲 Surprise"
    }
    
    for i, (emotion, label) in enumerate(emotions_display.items()):
        with cols[i]:
            st.markdown(
                f'<div style="background-color: {EMOTION_COLORS[emotion]}; '
                f'padding: 10px; border-radius: 5px; text-align: center; '
                f'color: {"white" if emotion in ["angry", "fear", "sad", "surprise"] else "black"}; '
                f'font-weight: bold;">{label}</div>',
                unsafe_allow_html=True
            )


# ============ STREAMLIT UI ============

st.title("Face Value: Movie Emotion Analysis")
st.markdown("""
Compare emotion recognition models on 50+ movies. 
**Face Value** (weak supervision) vs **RAF-DB** (standard db)
""")

# Display color legend at top
display_emotion_legend()
st.markdown("---")

tab1, tab2 = st.tabs(["📊 Movie Comparison", "📈 Timeline Analysis"])
with tab1:
    # Sidebar controls
    st.sidebar.header("Movie Comparison Controls")

    # Model selection
    model_choice = st.sidebar.radio(
        "Model(s) to display:",
        ["Face Value", "RAF-DB", "Side-by-side"],
        help="View one model or compare both"
    )

    # Movie group selection
    st.sidebar.subheader("Movie Selection")

    group_choice = st.sidebar.selectbox(
        "Choose a preset group:",
        ["Custom"] + list(MOVIE_GROUPS.keys()),
        index=1  # Default to "Featured Selection"
    )

    # Get initial movie list based on group
    if group_choice == "Custom":
        initial_movies = MOVIE_GROUPS["Featured Selection"][:5]  # Start with 5
    else:
        initial_movies = MOVIE_GROUPS[group_choice][:10]  # Max 10

    # Movie multiselect
    all_movies = sorted(fv_tidy['movie'].to_list())
    selected_movies = st.sidebar.multiselect(
        "Select movies (max 10):",
        all_movies,
        default=initial_movies,
        max_selections=10,
        format_func=lambda x: x.replace("_", " ").title()
    )

    # Validation
    if not selected_movies:
        st.warning("⚠️ Please select at least one movie")
        st.stop()

    # ============ DISPLAY PLOTS ============

    if model_choice == "Side-by-side":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Face Value (weak supervision)")
            fv_tidy = prepare_stacked_data(fv_tidy, selected_movies)
            fig_fv = create_stacked_bar(
                fv_tidy, 
                "Face Value: Emotion Distribution by Movie"
            )
            st.plotly_chart(fig_fv, use_container_width=True)
        
        with col2:
            st.subheader("RAF-DB (benchmark dataset)")
            raf_tidy = prepare_stacked_data(raf_tidy, selected_movies)
            fig_raf = create_stacked_bar(
                raf_tidy,
                "RAF-DB: Emotion Distribution by Movie"
            )
            st.plotly_chart(fig_raf, use_container_width=True)

    elif model_choice == "Face Value":
        fv_tidy = prepare_stacked_data(fv_tidy, selected_movies)
        fig = create_stacked_bar(
            fv_tidy,
            "Face Value: Emotion Distribution by Movie (Stacked %)"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # RAF-DB
        raf_tidy = prepare_stacked_data(raf_tidy, selected_movies)
        fig = create_stacked_bar(
            raf_tidy,
            "RAF-DB: Emotion Distribution by Movie (Stacked %)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ============ INSIGHTS ============

    st.markdown("---")
    st.subheader("💡 What to Notice")

    if group_choice == "Comedies":
        st.info("""
        **Comedies** should show higher happy proportions.
        - Face Value: Detects happy more often in comedies
        - RAF-DB: Defaults to surprise regardless of genre
        """)
    elif group_choice == "Kids Movies":
        st.info("""
        **Kids movies** typically have upbeat emotional content.
        - Face Value: Shows more happy and surprise
        - RAF-DB: Misses genre-appropriate emotional signals
        """)
    elif group_choice == "Dark Dramas":
        st.info("""
        **Dark dramas** should show more sad/fear emotions.
        - Face Value: Captures appropriate negative emotions
        - RAF-DB: Still dominated by surprise
        """)
    else:
        st.info("""
        Compare the emotional diversity:
        - **Face Value**: Shows variety across all 5 emotions
        - **RAF-DB**: Dominated by surprise across most movies
        """)

# ============ TAB 2: TIMELINE ANALYSIS ============

with tab2:
    st.sidebar.header("Timeline Controls")
    
    timeline_model = st.sidebar.radio(
        "Model(s) to display:",
        ["Face Value", "RAF-DB", "Side-by-side"],
        key="timeline_model"
    )
    
    timeline_movie = st.sidebar.selectbox(
        "Select movie:",
        sorted(fv_tidy['movie'].to_list()),
        format_func=lambda x: x.replace("_", " ").title(),
        key="timeline_movie"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Minimum confidence:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Filter predictions below this confidence level"
    )
    
    # Generate timeline data
    if timeline_model == "Side-by-side":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Face Value (weak supervision)")
            fv_timeline = prepare_timeline_data(fv_time, timeline_movie, confidence_threshold)
            if fv_timeline is not None:
                fig_fv = create_timeline_plot(fv_timeline, timeline_movie, confidence_threshold)
                st.plotly_chart(fig_fv, use_container_width=True)
            else:
                st.warning("No data available for this movie at selected confidence level")
        
        with col2:
            st.subheader("RAF-DB (benchmark dataset)")
            raf_timeline = prepare_timeline_data(raf_time, timeline_movie, confidence_threshold)
            if raf_timeline is not None:
                fig_raf = create_timeline_plot(raf_timeline, timeline_movie, confidence_threshold)
                st.plotly_chart(fig_raf, use_container_width=True)
            else:
                st.warning("No data available for this movie at selected confidence level")
    
    elif timeline_model == "Face Value":
        fv_timeline = prepare_timeline_data(fv_time, timeline_movie, confidence_threshold)
        if fv_timeline is not None:
            fig = create_timeline_plot(fv_timeline, timeline_movie, confidence_threshold)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for this movie at selected confidence level")
    
    else:  # RAF-DB
        raf_timeline = prepare_timeline_data(raf_time, timeline_movie, confidence_threshold)
        if raf_timeline is not None:
            fig = create_timeline_plot(raf_timeline, timeline_movie, confidence_threshold)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for this movie at selected confidence level")
    
    # Key moments (optional - you can add specific movie annotations here)
    st.markdown("---")
    st.subheader("🎬 Key Observations")
    
    key_moments = {
        "finding_nemo": "Reunion at ~84 minutes - watch for happy surge",
        "real_steel": "Redemption arc begins ~90 minutes",
        "inside_out": "Joy leaves headquarters ~27 minutes",
        "dark_knight": "Joker interrogation scene ~80 minutes",
    }
    
    if timeline_movie in key_moments:
        st.info(f"**{timeline_movie.replace('_', ' ').title()}**: {key_moments[timeline_movie]}")


# Footer
st.markdown("---")
st.markdown("""
📊 [View on GitHub](https://github.com/yourusername/face-value) | 
📝 [Read the full analysis](https://pixelprocess.org)
""")