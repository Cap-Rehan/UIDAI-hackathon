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
# # Update-Heavy but Enrolment-Light Regions

# %%
from book1 import district_df

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% [markdown]
# ## Compute Update Pressure Metrics

# %%
district_df["total_updates"] = (
    district_df["demo_activity"] +
    district_df["bio_activity"]
)

district_df["update_to_enrolment_ratio"] = (
    district_df["total_updates"] /
    district_df["total_enrolments"].replace(0, np.nan)
)

# %%
district_df[
    ["state", "district", "total_enrolments", "total_updates", "update_to_enrolment_ratio"]
].describe()

# %% [markdown]
# ## Identify Update-Heavy

# %%
ratio_threshold = district_df["update_to_enrolment_ratio"].quantile(0.90)

update_heavy = district_df[
    district_df["update_to_enrolment_ratio"] >= ratio_threshold
].sort_values("update_to_enrolment_ratio", ascending=False)

# %%
update_heavy.head(10)

# %% [markdown]
# ## Visualization: Ratio-Based Stress

# %%
sns.set_context("talk")
sns.set_style("white")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 227

# Fix the Warning: Use .copy() to create a standalone table
top10_ratio = update_heavy.head(10).copy()

# This fixes "Medchal?malkajgiri" -> "Medchal-Malkajgiri"
if 'district' in top10_ratio.columns:
    top10_ratio['district'] = top10_ratio['district'].astype(str).str.replace('?', '-')

# Create the Plot
ax = sns.barplot(
    data=top10_ratio,
    y="district",
    x="update_to_enrolment_ratio",
    hue="state",
    palette="rocket",
    dodge=False
)

# --- CENTER ALIGNED TITLES ---
plt.figtext(0.5, 0.93, "Top 10 Districts: Update-to-Enrolment Pressure", 
            fontsize=24, weight='bold', ha='center')

plt.figtext(0.5, 0.88, "Ratio of updates per new enrolment (Higher = More maintenance load)", 
            fontsize=14, color='#666666', ha='center')

# Add Data Labels (White text inside bars)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=-50, fontsize=12, color='white', weight='bold')

# Clean Axes
plt.xlabel("")
plt.ylabel("")
plt.xticks([]) 
sns.despine(left=True, bottom=True)

# --- LEGEND LOWER RIGHT ---
sns.move_legend(
    ax, "lower right",
    bbox_to_anchor=(1, 0), 
    title="",
    frameon=False,
)

# THE LAYOUT FIX
plt.tight_layout(rect=[0, 0, 1, 0.85])
plt.show()

# %% [markdown]
# ## Contrast Check

# %%
district_df.sort_values(
    "total_enrolments", ascending=False
)[
    ["state", "district", "total_enrolments", "update_to_enrolment_ratio"]
].head(10)

# %% [markdown]
# ## Insight: Update-Heavy but Enrolment-Light Regions
#
# ### Several districts exhibit update activity that is disproportionately high relative to their enrolment volume.
#
# This indicates that operational load in these regions is driven primarily by
# identity maintenance rather than new enrolments.
#
# Such districts require:
# - Update-focused infrastructure planning
# - Capacity allocation based on lifecycle load, not population size
#
# This insight is derived entirely from relative, aggregated metrics and avoids
# assumptions about individual behavior.
