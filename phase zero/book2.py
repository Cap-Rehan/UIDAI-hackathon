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
from book1 import pincode_df

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% [markdown]
# ## Re-aggregate to District Level & Filter Noise

# %%
# 1. Re-aggregate Pincode data back to District Level for Regional Analysis
region_df = pincode_df.groupby(['state', 'district'])[[
    'total_enrolments', 'demo_activity', 'bio_activity'
]].sum().reset_index()

# 2. Calculate Total Activity
region_df["total_activity"] = (
    region_df["total_enrolments"] +
    region_df["demo_activity"] +
    region_df["bio_activity"]
)

# 3. Apply Minimum Volume Filter (Fixes "Small Number Noise")
# We only analyze districts with significant activity (> 1000 transactions)
VOLUME_THRESHOLD = 1000
region_df = region_df[region_df["total_activity"] > VOLUME_THRESHOLD].copy()

# %% [markdown]
# ## Compute Split Update Ratios (Bio vs Demo)

# %%
# We calculate separate ratios to distinguish infrastructure needs
# Bio Ratio -> Need for Iris/Fingerprint Scanners
# Demo Ratio -> Need for Data Entry Terminals

region_df["bio_to_enrol_ratio"] = (
    region_df["bio_activity"] / region_df["total_enrolments"].replace(0, np.nan)
)

region_df["demo_to_enrol_ratio"] = (
    region_df["demo_activity"] / region_df["total_enrolments"].replace(0, np.nan)
)

# Total Maintenance Ratio (for sorting)
region_df["total_maintenance_ratio"] = region_df["bio_to_enrol_ratio"] + region_df["demo_to_enrol_ratio"]

# %%
region_df[
    ["state", "district", "total_enrolments", "bio_to_enrol_ratio", "demo_to_enrol_ratio"]
].describe()

# %% [markdown]
# ## Identify Update-Heavy

# %%
ratio_threshold = region_df["total_maintenance_ratio"].quantile(0.90)

update_heavy = region_df[
    region_df["total_maintenance_ratio"] >= ratio_threshold
].sort_values("total_maintenance_ratio", ascending=False)

# %%
update_heavy.head(10)

# %% [markdown]
# ## Visualization: Maintenance-Heavy Districts by Dominant Need

# %%
sns.set_context("talk")
sns.set_style("white")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 227

# Fix the Warning: Use .copy() to create a standalone table
top10_maintenance = update_heavy.head(10).copy()

# Determine Dominant Need for coloring
def get_dominant_need(row):
    if row['bio_to_enrol_ratio'] > row['demo_to_enrol_ratio']:
        return 'Biometric Heavy (Scanners Needed)'
    else:
        return 'Demographic Heavy (Data Entry Needed)'

top10_maintenance['dominant_need'] = top10_maintenance.apply(get_dominant_need, axis=1)

# This fixes "Medchal?malkajgiri" -> "Medchal-Malkajgiri"
if 'district' in top10_maintenance.columns:
    top10_maintenance['district'] = top10_maintenance['district'].astype(str).str.replace('?', '-')

# Create the Plot
ax = sns.barplot(
    data=top10_maintenance,
    y="district",
    x="total_maintenance_ratio",
    hue="dominant_need",
    palette={"Biometric Heavy (Scanners Needed)": "#e74c3c", "Demographic Heavy (Data Entry Needed)": "#3498db"},
    dodge=False
)

# --- CENTER ALIGNED TITLES ---
plt.figtext(0.5, 0.93, "Top 10 Districts: High Maintenance Pressure", 
            fontsize=24, weight='bold', ha='center')

plt.figtext(0.5, 0.88, "Districts categorized by dominant infrastructure need (Bio vs Demo)", 
            fontsize=14, color='#666666', ha='center')

# --- LEGEND LOWER RIGHT ---
sns.move_legend(
    ax, "lower right",
    bbox_to_anchor=(1, 0), 
    title="",
    frameon=False,
)

# Clean Axes
plt.xlabel("")
plt.ylabel("")
plt.xticks([]) 
sns.despine(left=True, bottom=True)

# THE LAYOUT FIX
plt.tight_layout(rect=[0, 0, 1, 0.85])
plt.show()

# %% [markdown]
# ## Contrast Check

# %%
# Comparing with high enrolment districts to see the difference
region_df.sort_values(
    "total_enrolments", ascending=False
)[
    ["state", "district", "total_enrolments", "total_maintenance_ratio"]
].head(10)

# %% [markdown]
# ## Insight: Update-Heavy but Enrolment-Light Regions
#
# ### Several districts exhibit update activity that is disproportionately high relative to their enrolment volume.
#
# By splitting the ratios, we can now distinguish the specific infrastructure need:
# - **Biometric Heavy:** Regions where older residents are updating biometrics (Requires Scanners).
# - **Demographic Heavy:** Regions with high migration or address changes (Requires Data Entry).
#
# This indicates that operational load in these regions is driven primarily by
# identity maintenance rather than new enrolments.
#
# Such districts require:
# - Update-focused infrastructure planning
# - Capacity allocation based on lifecycle load, not population size
#
# *Note: We applied a volume filter (>1000) to ensure these are significant operational centers, not statistical noise.*
#
# This insight is derived entirely from relative, aggregated metrics and avoids
# assumptions about individual behavior.