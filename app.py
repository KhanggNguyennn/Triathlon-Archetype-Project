import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Triathlete Archetypes — Nice 2026",
    page_icon="🏊🚴‍♂️🏃",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [class*="st-"], [data-testid="stAppViewContainer"] {
        font-family: "Times New Roman", Times, serif;
    }
    button, input, textarea, select {
        font-family: "Times New Roman", Times, serif !important;
    }
    [data-testid="stIconMaterial"], .material-symbols-rounded {
        font-family: "Material Symbols Rounded" !important;
        font-weight: normal !important;
        font-style: normal !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TIME_COLUMNS = ["Swim (Seconds)", "Bike (Seconds)", "Run (Seconds)"]
STRENGTH_COLUMNS = ["Swim Strength", "Bike Strength", "Run Strength"]
PROFILE_COLUMNS = ["Swim Profile", "Bike Profile", "Run Profile"]

COLOR_MAP = {
    "Balanced": "#06D6A0",
    "Run-oriented": "#EF476F",
    "Swim-oriented / Bike-weak": "#118AB2",
    "Front-loaded / Run-weak": "#7B2CBF",
}


def name_clusters(centers: pd.DataFrame) -> dict[int, str]:
    """Name clusters from their centroids instead of trusting arbitrary IDs."""
    remaining = set(centers.index)
    mapping: dict[int, str] = {}

    balanced = centers.abs().sum(axis=1).idxmin()
    mapping[balanced] = "Balanced"
    remaining.remove(balanced)

    run_oriented = centers.loc[list(remaining), "Run Strength"].idxmax()
    mapping[run_oriented] = "Run-oriented"
    remaining.remove(run_oriented)

    run_weak = centers.loc[list(remaining), "Run Strength"].idxmin()
    mapping[run_weak] = "Front-loaded / Run-weak"
    remaining.remove(run_weak)

    for cluster in remaining:
        mapping[cluster] = "Swim-oriented / Bike-weak"

    return mapping


@st.cache_data(show_spinner="Analyzing race results…")
def run_analysis(file_bytes: bytes, k: int = 4):
    df = pd.read_csv(io.BytesIO(file_bytes))

    required = {"Athlete Name", "Status", "Overall Rank", *TIME_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    valid_mask = (df["Status"] == "FIN") & df[TIME_COLUMNS].notna().all(axis=1)
    athletes = df.loc[
        valid_mask, ["Athlete Name", "Overall Rank", *TIME_COLUMNS]
    ].copy()
    athletes.reset_index(drop=True, inplace=True)

    scaled = pd.DataFrame(
        StandardScaler().fit_transform(athletes[TIME_COLUMNS]),
        columns=TIME_COLUMNS,
        index=athletes.index,
    )

    # Remove overall ability and retain each athlete's relative discipline profile.
    athlete_average = scaled.mean(axis=1)
    strengths = scaled.sub(athlete_average, axis=0).mul(-1)
    strengths.columns = STRENGTH_COLUMNS

    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    athletes["Cluster"] = model.fit_predict(strengths)
    athletes[STRENGTH_COLUMNS] = strengths

    centers = pd.DataFrame(model.cluster_centers_, columns=STRENGTH_COLUMNS)
    cluster_names = name_clusters(centers)
    athletes["Archetype"] = athletes["Cluster"].map(cluster_names)

    # Softmax produces positive shares that add to 100 for the ternary chart.
    exp_strengths = np.exp(strengths.to_numpy())
    profiles = exp_strengths / exp_strengths.sum(axis=1, keepdims=True) * 100
    athletes[PROFILE_COLUMNS] = profiles

    # The source uses large sentinel values such as 99999 for unranked athletes.
    ranked = athletes[
        pd.to_numeric(athletes["Overall Rank"], errors="coerce").between(
            1, len(athletes)
        )
    ].copy()

    cutoff = ranked["Overall Rank"].quantile(0.10)
    summary = ranked.groupby("Archetype").agg(
        Athletes=("Overall Rank", "count"),
        Median_Rank=("Overall Rank", "median"),
        Mean_Rank=("Overall Rank", "mean"),
        Q1_Rank=("Overall Rank", lambda x: x.quantile(0.25)),
        Q3_Rank=("Overall Rank", lambda x: x.quantile(0.75)),
    )
    summary["Rank_IQR"] = summary["Q3_Rank"] - summary["Q1_Rank"]
    top_10 = ranked[ranked["Overall Rank"] <= cutoff]
    summary["Top_10_Percent"] = (
        ranked.assign(Top_10=ranked["Overall Rank"] <= cutoff)
        .groupby("Archetype")["Top_10"]
        .mean()
        .mul(100)
    )
    summary["Top_10_Share"] = (
        top_10["Archetype"].value_counts(normalize=True).mul(100).reindex(summary.index, fill_value=0)
    )
    summary = summary.sort_values("Median_Rank").reset_index()

    return df, athletes, ranked, centers, summary, cutoff


def ternary_chart(athletes: pd.DataFrame, selected_name: str | None):
    fig = px.scatter_ternary(
        athletes,
        a="Swim Profile",
        b="Bike Profile",
        c="Run Profile",
        color="Archetype",
        hover_name="Athlete Name",
        hover_data={
            "Overall Rank": True,
            "Swim Profile": ":.1f",
            "Bike Profile": ":.1f",
            "Run Profile": ":.1f",
            "Cluster": False,
        },
        color_discrete_map=COLOR_MAP,
    )
    fig.update_traces(
        marker={"size": 6, "opacity": 0.55, "line": {"width": 0.3, "color": "white"}}
    )

    if selected_name:
        selected = athletes[athletes["Athlete Name"] == selected_name]
        fig.add_trace(
            go.Scatterternary(
                a=selected["Swim Profile"],
                b=selected["Bike Profile"],
                c=selected["Run Profile"],
                mode="markers+text",
                text=selected["Athlete Name"],
                textposition="top center",
                marker={
                    "size": 15,
                    "color": "black",
                    "symbol": "diamond",
                    "line": {"width": 2, "color": "white"},
                },
                name="Selected athlete",
                hovertemplate=(
                    "<b>%{text}</b><br>Swim profile: %{a:.1f}<br>"
                    "Bike profile: %{b:.1f}<br>Run profile: %{c:.1f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=720,
        margin={"l": 35, "r": 35, "t": 20, "b": 35},
        legend_title_text="Athlete archetype",
        font={"family": "Times New Roman, Times, serif"},
    )
    fig.update_ternaries(
        aaxis_title="Swim orientation",
        baxis_title="Bike orientation",
        caxis_title="Run orientation",
    )
    return fig


st.title("Triathlete Performance Archetypes")
st.caption("""Khang Nguyen""")

intro_column, photo_column = st.columns([2, 1], vertical_alignment="top")

with intro_column:
    st.markdown(
        """
This project uses **K-means clustering** to identify athletes with similar relative
swim, bike, and run profiles from the IRONMAN 70.3 World Championship in Nice, then examines how those profiles are associated with
finishing rank. The central question is whether one strong discipline can compensate
for a major weakness at a world-championship race.
"""
    )

    with st.expander("Why I chose this project", expanded=True):
        st.markdown(
            """
As a triathlon fan who is now training for the sport myself, I wanted to understand
how successful athletes balance the swim, bike, and run. That interest became more
personal after I volunteered at IRONMAN 70.3 NICE World Championship 2026 event and experienced the demands of the
sport up close.

This led me to examine athletes competing at the 2026 IRONMAN 70.3 World Championship
in Nice—one of the sport's most competitive stages. By studying the performance
profiles of this field, I aim to explore whether competitive success is more strongly
associated with specialization in one discipline or with avoiding major weaknesses
across all three. The findings cannot determine an ideal training plan by themselves,
but they can provide evidence to inform how athletes think about race preparation and
discipline balance.
"""
        )

with photo_column:
    st.image("image.jpeg", caption="Jamie Riddle - Ranked 7th", width="stretch")

default_csv = Path("nice_ironman_results_2026.csv")
if default_csv.exists():
    csv_bytes = default_csv.read_bytes()
else:
    st.info(
        "The file `nice_ironman_results_2026.csv` must be in the same folder as `app.py`."
    )
    st.stop()

try:
    df, athletes, ranked, centers, summary, top_10_cutoff = run_analysis(csv_bytes)
except Exception as error:
    st.error(f"The data could not be analyzed: {error}")
    st.stop()

raw_count = len(df)
complete_count = len(athletes)
excluded_count = raw_count - complete_count
unranked_count = complete_count - len(ranked)

metric_cols = st.columns(4)
metric_cols[0].metric("Raw entries", f"{raw_count:,}")
metric_cols[1].metric("Complete finishers", f"{complete_count:,}")
metric_cols[2].metric("Incomplete / non-finishers", f"{excluded_count:,}")
metric_cols[3].metric("Complete but unranked", f"{unranked_count:,}")

tab_findings, tab_explore, tab_method, tab_data = st.tabs(
    ["Key findings", "Explore athletes", "Method", "Data"]
)

with tab_findings:
    st.header("What the race suggests")
    balanced = summary[summary["Archetype"] == "Balanced"]
    others = summary[summary["Archetype"] != "Balanced"]
    if not balanced.empty and not others.empty:
        balanced_share = balanced["Top_10_Share"].iloc[0]
        next_best_share = others["Top_10_Share"].max()
        balanced_rate = balanced["Top_10_Percent"].iloc[0]
        next_best_rate = others["Top_10_Percent"].max()
        ratio = balanced_rate / next_best_rate if next_best_rate else np.nan
        st.success(
            "At Nice 2026, top finishers were disproportionately characterized by "
            "balanced swim, bike, and run profiles. At this championship level, one "
            "pronounced strength did not appear to compensate for a major relative weakness."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Top 10% Balanced archetype", f"{balanced_share:.1f}%")
        c2.metric("Next-best archetype share", f"{next_best_share:.1f}%")
        c3.metric("Top-10 qualification rate: Balanced vs Run-oriented", f"{ratio:.1f}×")

    display_summary = summary.rename(
        columns={
            "Median_Rank": "Median rank",
            "Mean_Rank": "Mean rank",
            "Q1_Rank": "Q1 rank",
            "Q3_Rank": "Q3 rank",
            "Rank_IQR": "Rank IQR",
            "Top_10_Percent": "Top 10% rate within archetype",
            "Top_10_Share": "Share of top 10%",
        }
    )
    st.dataframe(
        display_summary.style.format(
            {
                "Median rank": "{:.0f}",
                "Mean rank": "{:.1f}",
                "Q1 rank": "{:.0f}",
                "Q3 rank": "{:.0f}",
                "Rank IQR": "{:.0f}",
                "Top 10% rate within archetype": "{:.1f}%",
                "Share of top 10%": "{:.1f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    rank_fig = px.box(
        ranked,
        x="Archetype",
        y="Overall Rank",
        color="Archetype",
        color_discrete_map=COLOR_MAP,
        points=False,
        category_orders={"Archetype": summary["Archetype"].tolist()},
        labels={"Overall Rank": "Overall rank (lower is better)"},
    )
    rank_fig.update_yaxes(autorange="reversed")
    rank_fig.update_layout(showlegend=False, template="plotly_white", height=480)
    rank_fig.update_layout(font={"family": "Times New Roman, Times, serif"})
    st.plotly_chart(rank_fig, width="stretch")

    st.caption(
        "This is an association within one race, not proof that adopting a balanced "
        "profile causes a better result."
    )

with tab_explore:
    st.header("Explore athlete profiles")
    athlete_options = sorted(athletes["Athlete Name"].dropna().unique())
    selected_name = st.selectbox(
        "Search for an athlete",
        options=[None, *athlete_options],
        format_func=lambda value: "No athlete selected" if value is None else value,
    )

    if selected_name:
        selected = athletes[athletes["Athlete Name"] == selected_name].iloc[0]
        cols = st.columns(5)
        cols[0].metric("Archetype", selected["Archetype"])
        cols[1].metric("Overall rank", f"{selected['Overall Rank']:,.0f}")
        cols[2].metric("Swim profile", f"{selected['Swim Profile']:.1f}")
        cols[3].metric("Bike profile", f"{selected['Bike Profile']:.1f}")
        cols[4].metric("Run profile", f"{selected['Run Profile']:.1f}")

    st.plotly_chart(
        ternary_chart(athletes, selected_name), width="stretch"
    )
    st.caption(
        "Profile values are visualization scores that sum to 100—not percentages of "
        "absolute athletic ability. A higher value means that discipline is stronger "
        "relative to the same athlete's other disciplines."
    )

    st.subheader("Successful specialists")
    specialists = ranked[ranked["Archetype"] != "Balanced"].nsmallest(
        20, "Overall Rank"
    )
    st.dataframe(
        specialists[
            ["Athlete Name", "Overall Rank", "Archetype", *STRENGTH_COLUMNS]
        ].style.format({column: "{:.2f}" for column in STRENGTH_COLUMNS}),
        width="stretch",
        hide_index=True,
    )

with tab_method:
    st.header("How the analysis works")
    st.markdown(
        """
1. Keep athletes marked **FIN** who have complete swim, bike, and run times.
2. Standardize each discipline so their different time scales are comparable.
3. Remove each athlete's average standardized performance. This isolates the
   **shape of the athlete's profile**, rather than overall speed.
4. Multiply by −1 so a positive score means a relative strength: lower time is better.
5. Use K-means with **K = 4** to group similar profiles.
6. Convert the three scores with softmax for display on a ternary chart.
"""
    )
    st.latex(r"s_{ij}=-\left(z_{ij}-\frac{1}{3}\sum_{d=1}^{3}z_{id}\right)")
    st.markdown(
        "Here, $z_{ij}$ is athlete *i*'s standardized time in discipline *j*. "
        "Positive $s_{ij}$ means a relative strength and negative means a relative weakness."
    )

    st.subheader("Cluster centers")
    centers_display = centers.copy()
    cluster_mapping = athletes[["Cluster", "Archetype"]].drop_duplicates()
    centers_display["Cluster"] = centers_display.index
    centers_display = centers_display.merge(cluster_mapping, on="Cluster")
    st.dataframe(
        centers_display[["Cluster", "Archetype", *STRENGTH_COLUMNS]].style.format(
            {column: "{:.3f}" for column in STRENGTH_COLUMNS}
        ),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Why K = 4?"):
        k_rows = []
        strengths = athletes[STRENGTH_COLUMNS]
        for k in range(2, 9):
            candidate = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = candidate.fit_predict(strengths)
            k_rows.append(
                {
                    "K": k,
                    "Inertia": candidate.inertia_,
                    "Silhouette score": silhouette_score(strengths, labels),
                }
            )
        k_results = pd.DataFrame(k_rows)
        st.dataframe(k_results, width="stretch", hide_index=True)
        st.caption(
            "The scores are diagnostic evidence, not a rule that automatically proves "
            "four is the only valid interpretation. Four also produced interpretable profiles."
        )

    st.subheader("Limitations")
    st.markdown(
        """
- This is one championship race, so course, weather, field quality, and race-day strategy matter.
- The analysis is observational and cannot establish causality.
- Relative strengths are not measures of absolute ability. A world-class run can still be an
  athlete's weakest discipline relative to their own swim and bike.
- Athletes with incomplete timing data, DNFs, and disqualifications are excluded, creating
  survivorship and missing-data bias.
- Transition times are not included in the performance profile.
- Pooling divisions and genders may hide important subgroup differences.
- K-means assumes compact clusters and forces every athlete into one group; archetypes are a
  useful simplification, not natural laws.
"""
    )

with tab_data:
    st.header("Clean analysis data")
    st.dataframe(athletes, width="stretch", hide_index=True)
    st.download_button(
        "Download processed athlete data",
        data=athletes.to_csv(index=False).encode("utf-8"),
        file_name="nice_2026_athlete_archetypes.csv",
        mime="text/csv",
    )
    with st.expander("Preview raw data"):
        st.dataframe(df.head(100), width="stretch", hide_index=True)

st.caption("Author: Khang Nguyen, student at Bocconi University-HEC Paris")

