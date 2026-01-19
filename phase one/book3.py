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
# OPTIMIZATION: Import raw dataframes from book1 to access Age Columns
# This avoids reloading files and ensures we use the same cleaned data.
from book1 import demo, bio

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 227
sns.set_style("white")

# %% [markdown]
# ## Aggregate Age Metrics (District Level)

# %%
# 1. Group Demo Activity by Age
demo_dist = demo.groupby(["state", "district"], as_index=False)[
    ["demo_age_5_17", "demo_age_17_"]
].sum()

# 2. Group Bio Activity by Age
bio_dist = bio.groupby(["state", "district"], as_index=False)[
    ["bio_age_5_17", "bio_age_17_"]
].sum()

# 3. Merge and Combine
district_df = demo_dist.merge(
    bio_dist, on=["state", "district"], suffixes=("_demo", "_bio")
)

# Calculate Total Activity by Age Group
district_df["activity_5_17"] = (
    district_df["demo_age_5_17"] + district_df["bio_age_5_17"]
)
district_df["activity_17_plus"] = (
    district_df["demo_age_17_"] + district_df["bio_age_17_"]
)

# Keep only relevant columns
district_df = district_df[["state", "district", "activity_5_17", "activity_17_plus"]].copy()

# %% [markdown]
# ## Core Metric: Age-Skew Ratio

# %%
district_df["total_update_activity"] = (
    district_df["activity_5_17"] +
    district_df["activity_17_plus"]
)

# Share of activity that is Adult (17+)
district_df["age_17_plus_share"] = (
    district_df["activity_17_plus"] /
    district_df["total_update_activity"].replace(0, np.nan)
)

# %%
# Filter for significant volume to avoid noise
MIN_ACTIVITY = district_df["total_update_activity"].quantile(0.75)
filtered = district_df[district_df["total_update_activity"] >= MIN_ACTIVITY].copy()

# %% [markdown]
# ## Identify Target Zones (Adult vs Child Heavy)

# %%
# 1. Adult Heavy (High 17+ Share) -> Needs Permanent Centers
top10_adult_heavy = filtered.sort_values("age_17_plus_share", ascending=False).head(10)

# 2. Child Heavy (Low 17+ Share) -> Needs School Camps
# 
top10_child_heavy = filtered.sort_values("age_17_plus_share", ascending=True).head(10)

# Calculate Median for Reference
median_val = district_df["age_17_plus_share"].median()

# %% [markdown]
# ## Visualization

# %%
sns.set_context("talk")
sns.set_style("white")

plt.figure(figsize=[12, 8], dpi=227)

# We plot Adult Heavy districts to highlight the "Center" pressure
plot_data = top10_adult_heavy.copy()

ax = sns.barplot(
    data=plot_data,
    y="district",
    x="age_17_plus_share",
    hue="state",
    palette="crest",
    dodge=False
)

# --- MEDIAN LINE ---
plt.axvline(
    median_val,
    color="#FF4B4B",
    linestyle="--",
    linewidth=2,
    alpha=0.8
)

plt.text(x=median_val + 0.02, y=-0.9, s=f"National Median: {median_val:.2f}", 
         color="#FF4B4B", weight="bold", ha="left", va="center")

# --- FORMATTING ---
plt.xlim(0, 1.15)

plt.figtext(0.5, 0.93, "Top Districts: Adult (17+) Update Concentration", 
            fontsize=24, weight='bold', ha='center')
plt.figtext(0.5, 0.88, "Share of total activity contributed by the 17+ age group", 
            fontsize=14, color='#666666', ha='center')

# Labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=5, fontsize=12, color='black', weight='bold')

# Cleanup
plt.xlabel("Share of 17+ Activity")
plt.ylabel("")
plt.xticks([]) 
sns.despine(left=True, bottom=True)

# Legend
sns.move_legend(
    ax, "lower right",
    bbox_to_anchor=(1, 0.05),
    title="",
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.85])
plt.show()

# %% [markdown]
# ## Insight: Age-Driven Service Pressure
#
# ### Variation in age demographics dictates the mode of service delivery.
#
# - **High Adult Share (>85%):** These districts (shown above) are dominated by adult identity maintenance. **Strategy:** Strengthen permanent Seva Kendras with extra counters.
#
# - **Low Adult Share (<60%):** (Not shown, but calculated) These districts have high child activity (5-17). **Strategy:** Deploy mobile School Camps to capture mandatory biometric updates efficiently.