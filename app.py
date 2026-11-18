import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Football Player Performance Analyzer", layout="wide")
sns.set_theme(style="whitegrid")

# ==========================================
# 1. LOAD & CACHE DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("data set/footballdata.csv")
    for col in ["Teams", "Seasons", "Players"]:
        df[col] = df[col].astype(str).str.strip()
        
    # Feature Engineering
    df["Goal_Contributions"] = df["Goals"] + df["Assists"]
    df["Goals_per_Match"] = np.where(df["Matches"] > 0, (df["Goals"] / df["Matches"]).round(2), 0)
    df["Assists_per_Match"] = np.where(df["Matches"] > 0, (df["Assists"] / df["Matches"]).round(2), 0)
    df["GA_per_Match"] = np.where(df["Matches"] > 0, (df["Goal_Contributions"] / df["Matches"]).round(2), 0)
    return df

df = load_data()

# ==========================================
# 2. SIDEBAR INTERACTIVE FILTERS
# ==========================================
st.sidebar.header("Filter Dataset")

# Team Filter
all_teams = sorted(df["Teams"].unique())
selected_teams = st.sidebar.multiselect("Select Teams (Leave empty for All)", options=all_teams, default=[])

# Season Filter
all_seasons = sorted(df["Seasons"].unique(), reverse=True)
selected_seasons = st.sidebar.multiselect("Select Seasons (Leave empty for All)", options=all_seasons, default=[])

# Player Search / Filter
all_players = sorted(df["Players"].unique())
selected_players = st.sidebar.multiselect("Select Specific Players (Optional)", options=all_players, default=[])

# Numeric Sliders
min_matches = st.sidebar.slider(
    "Minimum Matches Played in a Season",
    min_value=int(df["Matches"].min()),
    max_value=int(df["Matches"].max()),
    value=10
)

min_rating = st.sidebar.slider(
    "Minimum Season Rating",
    min_value=float(df["Seasons Ratings"].min()),
    max_value=float(df["Seasons Ratings"].max()),
    value=float(df["Seasons Ratings"].min()),
    step=0.1
)

# Apply all active filters dynamically
filtered_df = df[
    (df["Matches"] >= min_matches) &
    (df["Seasons Ratings"] >= min_rating)
]

if selected_teams:
    filtered_df = filtered_df[filtered_df["Teams"].isin(selected_teams)]
if selected_seasons:
    filtered_df = filtered_df[filtered_df["Seasons"].isin(selected_seasons)]
if selected_players:
    filtered_df = filtered_df[filtered_df["Players"].isin(selected_players)]

# ==========================================
# 3. DASHBOARD HEADER & KPI SUMMARY CARDS
# ==========================================
st.title("Football Match & Player Performance Analyzer")

if filtered_df.empty:
    st.warning("No records match your current filter selection. Try relaxing the sliders or clearing filters in the sidebar.")
    st.stop()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Filtered Records", len(filtered_df))
col2.metric("Unique Players", filtered_df["Players"].nunique())
col3.metric("Total Goals", int(filtered_df["Goals"].sum()))
col4.metric("Total Assists", int(filtered_df["Assists"].sum()))
col5.metric("Avg Season Rating", f"{filtered_df['Seasons Ratings'].mean():.2f}")

st.divider()

# ==========================================
# 4. INTERACTIVE CHARTS
# ==========================================
left_chart, right_chart = st.columns(2)

with left_chart:
    st.subheader("Top Performers by Goal Contributions (G+A)")
    top_players = (
        filtered_df.groupby("Players", as_index=False)[["Goals", "Assists", "Goal_Contributions"]]
        .sum()
        .sort_values(by="Goal_Contributions", ascending=False)
        .head(10)
        .sort_values(by="Goal_Contributions", ascending=True)
    )
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.barh(top_players["Players"], top_players["Goals"], label="Goals", color="#1f77b4")
    ax1.barh(top_players["Players"], top_players["Assists"], left=top_players["Goals"], label="Assists", color="#ff7f0e")
    ax1.set_xlabel("Total Goals + Assists")
    ax1.legend()
    st.pyplot(fig1)

with right_chart:
    st.subheader("Matches Played vs. Goal Contributions")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    scatter = ax2.scatter(
        filtered_df["Matches"],
        filtered_df["Goal_Contributions"],
        c=filtered_df["Seasons Ratings"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="k",
        linewidth=0.4
    )
    plt.colorbar(scatter, ax=ax2, label="Season Rating")
    ax2.set_xlabel("Matches Played")
    ax2.set_ylabel("Goals + Assists")
    st.pyplot(fig2)

# ==========================================
# 5. HEAD-TO-HEAD PLAYER COMPARISON
# ==========================================
st.divider()
st.subheader("Head-to-Head Player Season Trajectory")

default_compare = [p for p in ["Lionel Messi", "Cristiano Ronaldo"] if p in all_players]
compare_players = st.multiselect(
    "Choose 2 or more players to compare across seasons:",
    options=all_players,
    default=default_compare if default_compare else all_players[:2]
)

if compare_players:
    comp_df = df[df["Players"].isin(compare_players)].sort_values(by="Seasons")
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=comp_df, x="Seasons", y="Goal_Contributions", hue="Players", marker="o", ax=ax3)
    ax3.set_ylabel("Goals + Assists per Season")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

# ==========================================
# 6. FILTERED DATA TABLE & CSV EXPORT
# ==========================================
st.divider()
st.subheader("Filtered Player Records")
st.dataframe(
    filtered_df.sort_values(by="Goal_Contributions", ascending=False),
    use_container_width=True
)

csv_export = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Results as CSV",
    data=csv_export,
    file_name="filtered_football_stats.csv",
    mime="text/csv"
)