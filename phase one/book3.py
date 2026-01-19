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
from adjustText import adjust_text

sns.set_context("talk")
sns.set_style("whitegrid")

plt.figure(figsize=[14, 7], dpi=227)

# 1. The "Norm" (Boxplot)
ax = sns.boxplot(
    x=filtered["age_17_plus_share"],
    color="#f0f2f5",
    width=0.4,
    linewidth=1.2,
    fliersize=0,
    boxprops=dict(alpha=0.8),
    whiskerprops=dict(color='#bdc3c7'),
    capprops=dict(color='#bdc3c7'),
    medianprops=dict(color='#7f8c8d', linewidth=2)
)

# 2. The "Hotspots" (Plot ALL top 10 points with UNIQUE colors)
sns.stripplot(
    data=top10_adult_heavy,
    x="age_17_plus_share",
    hue="district",     # <--- This assigns a different color to each district
    palette="tab10",    # <--- A standard, high-contrast palette
    size=13,
    jitter=False,
    alpha=0.9,
    edgecolor="white",
    linewidth=1.2,
    ax=ax,
    zorder=5,
    legend=False        # <--- Hide legend since we label the dots directly
)

# 3. Smart Annotation (Label ONLY the Top 5 to prevent clutter)
texts = []
# We only iterate through the first 5 rows for text labels
for i, (idx, row) in enumerate(top10_adult_heavy.head(5).iterrows()):
    
    # We preset a slight vertical offset (alternating up/down)
    start_y = 0.15 if i % 2 == 0 else -0.15
    
    t = plt.text(
        x=row["age_17_plus_share"],
        y=start_y, 
        s=f"{row['district']}\n({row['age_17_plus_share']:.2f})",
        color="#2c3e50", # Dark blue-grey for text (easier to read than multi-colored text)
        fontsize=11,
        fontweight='bold',
        ha='center',
        va='center'
    )
    texts.append(t)

# 4. Run adjust_text
adjust_text(texts,
            x=top10_adult_heavy.head(5)["age_17_plus_share"],
            y=[0] * 5, 
            force_points=0.5,
            force_text=0.6,
            expand_points=(1.2, 1.5), 
            expand_text=(1.1, 1.2),
            arrowprops=dict(arrowstyle="-", color='gray', lw=0.8, alpha=0.5)
            )

# 5. Contextual Lines
median_val = filtered["age_17_plus_share"].median()
plt.axvline(median_val, color="#95a5a6", linestyle="--", linewidth=1.5, zorder=0)
plt.text(median_val, 0.45, f"National Median\n({median_val:.2f})",
         color="#7f8c8d", fontsize=11, ha="center", va="bottom", backgroundcolor='white')

# 6. Titles & Layout
plt.figtext(0.5, 1.02, "Age-Driven Service Pressure: Distribution & Outliers",
            fontsize=24, weight='bold', ha='center')

plt.figtext(0.5, 0.97, "Districts with exceptionally high adult (17+) activity share require distinct infrastructure.",
            fontsize=14, color='#666666', ha='center')

plt.xlabel("Share of Update Activity from Adults (17+)", labelpad=15, weight='bold', fontsize=14)
plt.yticks([]) 
sns.despine(left=True) 

# Zoom in on the relevant part of the distribution
plt.xlim(0.45, 1.02)
plt.ylim(-0.5, 0.5)

plt.subplots_adjust(top=0.85, bottom=0.15)
plt.show()

# %% [markdown]
# ## Insight: Age-Driven Service Pressure
#
# ### The boxplot reveals that while most districts maintain a balance (Median ~0.70), extreme outliers exist where service demand is 95%+ adult-driven.
#
# **Why this visual matters:**
# By plotting the distribution, we prove that the highlighted districts (Red) are **statistically distinct** from the norm. They are not just "high"—they are structural anomalies.
#
# **Operational Strategy:**
# - **For the Outliers (Red Dots):** These districts require **permanent infrastructure** (Seva Kendras) with extra counters, as adult updates (biometrics) are time-consuming and cannot be done in camps.
# - **For the Norm (Grey Box):** Standard mix of permanent centers and periodic camps is sufficient.

# %% [markdown]
#
