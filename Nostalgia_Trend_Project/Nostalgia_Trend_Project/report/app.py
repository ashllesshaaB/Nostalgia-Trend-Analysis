import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Nostalgia Trend Analysis",
    layout="wide"
)

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/final_ai_trend_analysis.csv")

df = load_data()

st.title("📊 Nostalgia Trend Analysis Dashboard")
st.markdown("Advanced Viral Spike Detection & Platform Trend Analysis")

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("🔎 Filter Data")

platform_filter = st.sidebar.multiselect(
    "Select Platform",
    options=df["Platform"].unique(),
    default=df["Platform"].unique()
)

region_filter = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

content_filter = st.sidebar.multiselect(
    "Select Content Type",
    options=df["Content_Type"].unique(),
    default=df["Content_Type"].unique()
)

hashtag_filter = st.sidebar.multiselect(
    "Select Hashtag",
    options=df["Hashtag"].unique(),
    default=df["Hashtag"].unique()
)

filtered_df = df[
    (df["Platform"].isin(platform_filter)) &
    (df["Region"].isin(region_filter)) &
    (df["Content_Type"].isin(content_filter)) &
    (df["Hashtag"].isin(hashtag_filter))
]

# -------------------------------
# KPIs
# -------------------------------
total_posts = len(filtered_df)
total_spikes = filtered_df["Advanced_Spike"].sum()

dominant_cluster = None
if "Cluster" in filtered_df.columns:
    dominant_cluster = filtered_df.groupby("Cluster")["Advanced_Spike"].sum().idxmax()

col1, col2, col3 = st.columns(3)
col1.metric("Total Posts", total_posts)
col2.metric("Total Viral Spikes", int(total_spikes))
col3.metric("Dominant Cluster", dominant_cluster)

st.divider()

# -------------------------------
# Viral Spikes by Cluster
# -------------------------------
if "Cluster" in filtered_df.columns:
    st.subheader("📌 Viral Spikes by Cluster")
    cluster_spikes = (
        filtered_df.groupby("Cluster")["Advanced_Spike"]
        .sum()
        .reset_index()
    )
    st.bar_chart(cluster_spikes.set_index("Cluster"))

    st.markdown(
        "🔎 Interpretation: This chart shows how viral spikes are distributed across semantic clusters."
    )

st.divider()

# -------------------------------
# Viral Spikes by Platform
# -------------------------------
st.subheader("📌 Viral Spikes by Platform")

platform_spikes = (
    filtered_df.groupby("Platform")["Advanced_Spike"]
    .sum()
    .reset_index()
)

st.bar_chart(platform_spikes.set_index("Platform"))

st.markdown(
    "🔎 Interpretation: Instagram shows the highest spike frequency compared to other platforms."
)

st.divider()

# -------------------------------
# Spike Rate by Platform
# -------------------------------
st.subheader("📌 Spike Rate by Platform")

platform_stats = filtered_df.groupby("Platform").agg(
    Total_Posts=("Post_ID", "count"),
    Total_Spikes=("Advanced_Spike", "sum")
).reset_index()

platform_stats["Spike_Rate"] = (
    platform_stats["Total_Spikes"] /
    platform_stats["Total_Posts"]
)

st.bar_chart(platform_stats.set_index("Platform")["Spike_Rate"])

st.markdown(
    "🔎 Interpretation: Spike Rate represents probability of virality per platform."
)

st.divider()

# -------------------------------
# Statistical Test Section
# -------------------------------
st.subheader("📊 Statistical Significance Test")

contingency_table = pd.crosstab(
    filtered_df["Platform"],
    filtered_df["Advanced_Spike"]
)

if contingency_table.shape[0] > 1:
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    colA, colB = st.columns(2)
    colA.metric("Chi-Square Value", round(chi2, 4))
    colB.metric("p-value", round(p, 5))

    if p < 0.05:
        st.success("Result is statistically significant (p < 0.05)")
    else:
        st.warning("Result is NOT statistically significant (p ≥ 0.05)")

    st.markdown(
        "🔎 Interpretation: Chi-square test checks if platform and virality are statistically dependent."
    )

st.divider()

# -------------------------------
# Interactive Data Selection
# -------------------------------
st.subheader("🗂 Interactive Dataset Explorer")

# Add checkbox column
filtered_df = filtered_df.copy()
filtered_df["Select"] = False

edited_df = st.data_editor(
    filtered_df,
    use_container_width=True,
    num_rows="dynamic"
)

selected_rows = edited_df[edited_df["Select"] == True]

st.markdown("### ✅ Selected Rows")
st.write(selected_rows)

# -------------------------------
# Download Selected Rows
# -------------------------------
if len(selected_rows) > 0:
    csv = selected_rows.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Selected Rows",
        data=csv,
        file_name="selected_data.csv",
        mime="text/csv"
    )
