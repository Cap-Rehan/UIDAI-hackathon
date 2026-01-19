# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Age-Driven Service Pressure

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 227
sns.set_style("whitegrid")

# %%
enrol= pd.read_parquet("/Users/mrehanansari/Documents/UIDAI/data/parquet/enrol_clean.parquet")
demo= pd.read_parquet("/Users/mrehanansari/Documents/UIDAI/data/parquet/demo_clean.parquet")
bio= pd.read_parquet("/Users/mrehanansari/Documents/UIDAI/data/parquet/bio_clean.parquet")

# %%
print("Enrol:", enrol.columns.tolist())
print("Demo :", demo.columns.tolist())
print("Bio  :", bio.columns.tolist())

# %%
enrol.drop(columns= ['Unnamed: 0'], inplace= True)
demo.drop(columns= ['Unnamed: 0'], inplace= True)
bio.drop(columns= ['Unnamed: 0'], inplace= True)

# %%
print("Enrol:", enrol.columns.tolist())
print("Demo :", demo.columns.tolist())
print("Bio  :", bio.columns.tolist())

# %%
demo["activity_5_17"] = demo["demo_age_5_17"]
demo["activity_17_plus"] = demo["demo_age_17_"]

bio["activity_5_17"] = bio["bio_age_5_17"]
bio["activity_17_plus"] = bio["bio_age_17_"]

# %%
demo_dist = (
    demo.groupby(["state", "district"], as_index=False)[
        ["activity_5_17", "activity_17_plus"]
    ].sum()
)

bio_dist = (
    bio.groupby(["state", "district"], as_index=False)[
        ["activity_5_17", "activity_17_plus"]
    ].sum()
)

# %%
district_df = (
    demo_dist
    .merge(bio_dist, on=["state", "district"], suffixes=("_demo", "_bio"))
)

district_df["activity_5_17"] = (
    district_df["activity_5_17_demo"] +
    district_df["activity_5_17_bio"]
)

district_df["activity_17_plus"] = (
    district_df["activity_17_plus_demo"] +
    district_df["activity_17_plus_bio"]
)

district_df = district_df[
    ["state", "district", "activity_5_17", "activity_17_plus"]
]

district_df.head()

# %% [markdown]
# ## Core Metric: Age-Skew Ratio

# %%
district_df["total_update_activity"] = (
    district_df["activity_5_17"] +
    district_df["activity_17_plus"]
)

district_df["age_17_plus_share"] = (
    district_df["activity_17_plus"] /
    district_df["total_update_activity"].replace(0, np.nan)
)

# %%
district_df["age_17_plus_share"].describe()

# %%
MIN_ACTIVITY = district_df["total_update_activity"].quantile(0.75)  # or 0.8

filtered = district_df[
    district_df["total_update_activity"] >= MIN_ACTIVITY
]

# %%
threshold = filtered["age_17_plus_share"].quantile(0.90)

age_skewed = (
    filtered[
        filtered["age_17_plus_share"] >= threshold
    ]
    .sort_values("age_17_plus_share", ascending=False)
)

# %%
age_skewed.head()

# %%
sns.set_context("talk")
sns.set_style("white")

# 1. Setup Figure
plt.figure(figsize=[12, 8], dpi=227)

# 2. PREPARE & SORT
top10_age = age_skewed.head(10).copy()
top10_age = top10_age.sort_values("age_17_plus_share", ascending=False)

# (Ensure you have your full dataframe 'district_df' or similar for this calc)
# If 'district_df' is not available, replace it with your full source dataframe
median_val = district_df["age_17_plus_share"].median()

# 3. PLOT
ax = sns.barplot(
    data=top10_age,
    y="district",
    x="age_17_plus_share",
    hue="state",
    palette="crest",
    dodge=False
)

# --- FIX 1: THE MEDIAN LINE & TEXT ---
plt.axvline(
    median_val,
    color="#FF4B4B",
    linestyle="--",
    linewidth=2,
    alpha=0.8
)

# CHANGED: y=-0.7 puts the text ABOVE the first bar (which is at y=0).
# This prevents it from overlapping the bar or the label.
plt.text(x=median_val + 0.02, y=-0.9, s=f"National Median: {median_val:.2f}", 
         color="#FF4B4B", weight="bold", ha="left", va="center")

# --- FIX 2: CREATE 'BREATHING ROOM' FOR LEGEND ---
plt.xlim(0, 1.25)

# 4. TITLES
plt.figtext(0.5, 0.93, "Top Districts: Adult (17+) Update Concentration", 
            fontsize=24, weight='bold', ha='center')
plt.figtext(0.5, 0.88, "Share of total activity contributed by the 17+ age group", 
            fontsize=14, color='#666666', ha='center')

# 5. DATA LABELS
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=-40, fontsize=12, color='white', weight='bold')

# 6. CLEANUP
plt.xlabel("")
plt.ylabel("")
plt.xticks([]) 
sns.despine(left=True, bottom=True)

# --- FIX 3: LEGEND PLACEMENT ---
sns.move_legend(
    ax, "lower right",
    bbox_to_anchor=(1, 0.05),
    title="",
    frameon=False,
)

# 7. LAYOUT
plt.tight_layout(rect=[0, 0, 1, 0.85])
plt.show()

# %% [markdown]
# ## Insight: Age-Driven Service Pressure
#
# While Aadhaar update activity is expected to be higher in the 17+ age group, district-level patterns reveal **significant variation in the degree of adult concentration**.
#
# Across all districts, the **national median share of 17+ update activity is approximately 70%**. However, among districts with **high total update volume**, several exhibit **substantially higher concentration (≈80–86%)**, indicating that identity maintenance activity in these regions is **structurally adult-centric** rather than evenly distributed across life stages.
#
# This pattern suggests that Aadhaar updates in such districts tend to occur **later and in more concentrated phases**, which has implications for:
# - timing of service demand,
# - staffing and capacity planning,
# - and targeted, age-aware outreach strategies.
#
# All findings are based on aggregated activity patterns and do not rely on individual-level assumptions or causal attribution.

# %% [markdown]
#
