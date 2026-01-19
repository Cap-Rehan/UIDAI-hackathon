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

# --- GLOBAL DATA CLEANING START ---
# Fix State Names (Title Case, Strip, and Specific Replacements)
region_df['state'] = region_df['state'].str.title().str.strip()
region_df['state'] = region_df['state'].replace({
    'Westbengal': 'West Bengal',
    'Daman And Diu': 'Daman & Diu', 
    'Dadra And Nagar Haveli': 'Dadra & Nagar Haveli',
    'Andaman And Nicobar Islands': 'A & N Islands'
})

# Fix District Typos Globally (e.g. Medchal?malkajgiri)
region_df['district'] = region_df['district'].astype(str).str.replace('?', '-')
# --- GLOBAL DATA CLEANING END ---

# 3. Apply Minimum Volume Filter (Fixes "Small Number Noise")
# We only analyze districts with significant activity (> 1000 transactions)
VOLUME_THRESHOLD = 1000
region_df = region_df[region_df["total_activity"] > VOLUME_THRESHOLD].copy()

# %%
print(f"Districts after filtering: {len(region_df)}")
region_df.head()

# %% [markdown]
# ## Compute Update Pressure Metrics

# %%
region_df["total_updates"] = (
    region_df["demo_activity"] +
    region_df["bio_activity"]
)

region_df["update_to_enrolment_ratio"] = (
    region_df["total_updates"] /
    region_df["total_enrolments"].replace(0, np.nan)
)

# %%
region_df[
    ["state", "district", "total_enrolments", "total_updates", "update_to_enrolment_ratio"]
].describe()

# %% [markdown]
# ## Identify Update-Heavy

# %%
ratio_threshold = region_df["update_to_enrolment_ratio"].quantile(0.90)

update_heavy = region_df[
    region_df["update_to_enrolment_ratio"] >= ratio_threshold
].sort_values("update_to_enrolment_ratio", ascending=False)

# %%
print(f"Update-heavy districts found: {len(update_heavy)}")
update_heavy.head(10)

# %% [markdown]
# ## Visualization 1: Ratio-Based Stress

# %%
import textwrap

# Fix the Warning: Use .copy() to create a standalone table
top10_ratio = update_heavy.head(10).copy()

# --- FIX 1: HANDLE LONG NAMES (Word Wrap) ---
# Wraps names longer than 20 characters
top10_ratio['district'] = top10_ratio['district'].apply(
    lambda x: textwrap.fill(x, 15) if len(x) > 20 else x
)

# Create new figure for Graph 1
fig1, ax1 = plt.subplots(figsize=(12, 8), dpi=150)

sns.set_context("talk")
sns.set_style("white")

# Create the Plot
sns.barplot(
    data=top10_ratio,
    y="district",
    x="update_to_enrolment_ratio",
    hue="state",
    palette="rocket",
    dodge=False,
    errorbar=None, # --- FIX 2: REMOVE THE BLACK LINE ---
    ax=ax1
)

# --- FIX 3: CREATE SPACE FOR LEGEND ---
# Extend x-axis to make room on the right
max_val = top10_ratio['update_to_enrolment_ratio'].max()
ax1.set_xlim(0, max_val * 0.8)

# Title and Subtitle
ax1.set_title("Top 10 Districts: Update-to-Enrolment Pressure\n", 
              fontsize=30, weight='bold', loc='left', x= -0.1)
ax1.text(0, 1.02, "Ratio of updates per new enrolment (Higher = More maintenance load)", 
         fontsize=12, color='#666666', ha='left', transform=ax1.transAxes)

# Add Data Labels (White text inside bars)
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.2f', padding=-50, fontsize=11, color='white', weight='bold')

# Clean Axes
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.set_xticks([])
sns.despine(left=True, bottom=True)

# Legend Fix: Move to lower right (now sits in the empty space)
sns.move_legend(
    ax1, "lower right",
    bbox_to_anchor=(1, 0),
    title="",
    frameon=False,
)

plt.tight_layout()
plt.show()

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

# %% [markdown]
# ---
# # Deeper Analysis: Split by Update Type (Bio vs Demo)
# ---

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
    ["state", "district", "total_enrolments", "bio_to_enrol_ratio", "demo_to_enrol_ratio", "total_maintenance_ratio"]
].describe()

# %% [markdown]
# ## Identify Maintenance-Heavy Districts

# %%
maintenance_threshold = region_df["total_maintenance_ratio"].quantile(0.90)

maintenance_heavy = region_df[
    region_df["total_maintenance_ratio"] >= maintenance_threshold
].sort_values("total_maintenance_ratio", ascending=False)

# %%
print(f"Maintenance-heavy districts found: {len(maintenance_heavy)}")
maintenance_heavy.head(10)

# %% [markdown]
# ## Visualization 2: Maintenance-Heavy Districts by Dominant Need

# %%
import textwrap

# Fix the Warning: Use .copy() to create a standalone table
top10_maintenance = maintenance_heavy.head(10).copy()

# --- FIX 1: HANDLE LONG NAMES (DISTRICTS) ---
# This wraps any name longer than 20 chars into multiple lines
top10_maintenance['district'] = top10_maintenance['district'].apply(
    lambda x: textwrap.fill(x, 15) if len(x) > 20 else x
)

# Determine Dominant Need for coloring
def get_dominant_need(row):
    if row['bio_to_enrol_ratio'] > row['demo_to_enrol_ratio']:
        return 'Bio-Heavy (Scanners Needed)'
    else:
        return 'Demo-Heavy (Data Entry Needed)'

top10_maintenance['dominant_need'] = top10_maintenance.apply(get_dominant_need, axis=1)

# Create new figure for Graph 2
fig2, ax2 = plt.subplots(figsize=(12, 8), dpi=150)

sns.set_context("talk")
sns.set_style("white")

# Create the Plot
sns.barplot(
    data=top10_maintenance,
    y="district",
    x="total_maintenance_ratio",
    hue="dominant_need",
    palette={"Bio-Heavy (Scanners Needed)": "#e74c3c", "Demo-Heavy (Data Entry Needed)": "#3498db"},
    dodge=False,
    errorbar=None,
    ax=ax2
)

# --- FIX 2: CREATE SPACE FOR LEGEND ---
# Get the maximum value and add padding to the right so the legend fits
max_val = top10_maintenance['total_maintenance_ratio'].max()
ax2.set_xlim(0, max_val * 0.8)

# Title and Subtitle
ax2.set_title("Top 10 Districts: High Maintenance Pressure\n", 
              fontsize=30, weight='bold', loc='left', x= -0.1)
ax2.text(0, 1.02, "Districts categorized by dominant infrastructure need (Bio vs Demo)", 
         fontsize=12, color='#666666', ha='left', transform=ax2.transAxes)

# Add Data Labels
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.2f', padding=-50, fontsize=11, color='white', weight='bold')

# Clean Axes
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.set_xticks([])
sns.despine(left=True, bottom=True)

# Legend Fix: Move to lower right
sns.move_legend(
    ax2, "lower right",
    bbox_to_anchor=(1, 0),
    title="",
    frameon=False,
)

# --- FIX 3: WRAP LEGEND TEXT ---
# Access the legend created above and wrap the text to 20 characters
for text in ax2.get_legend().get_texts():
    text.set_text(textwrap.fill(text.get_text(), 20))

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Insight: Maintenance-Heavy Districts by Dominant Need
#
# ### By splitting the update ratios, we can now distinguish the specific infrastructure need for each district.
#
# The analysis reveals two distinct categories:
# - **Biometric Heavy (Red):** Regions where residents are primarily updating biometrics (fingerprints, iris). These require more Iris/Fingerprint Scanners.
# - **Demographic Heavy (Blue):** Regions with high address/name changes (likely due to migration). These require more Data Entry Terminals.
#
# This distinction enables UIDAI to:
# - Allocate equipment budgets more precisely
# - Deploy the RIGHT type of infrastructure to each district
# - Avoid wasteful spending on wrong equipment
#
# *Note: Volume filter (>1000) ensures these are operationally significant districts, not statistical noise.*

# %% [markdown]
# ---

# %% [markdown]
# ## Contrast Check

# %%
# Comparing with high enrolment districts to see the difference
region_df.sort_values(
    "total_enrolments", ascending=False
)[
    ["state", "district", "total_enrolments", "update_to_enrolment_ratio", "total_maintenance_ratio"]
].head(10)

# %% [markdown]
# ## Combined Insight: Update-Heavy but Enrolment-Light Regions
#
# ### Several districts exhibit update activity that is disproportionately high relative to their enrolment volume.
#
# **Graph 1 (Update-to-Enrolment Pressure)** shows us WHICH districts have high update pressure overall.
#
# **Graph 2 (Maintenance by Dominant Need)** tells us WHAT TYPE of updates are driving that pressure.
#
# Together, these insights enable:
# - Update-focused infrastructure planning
# - Capacity allocation based on lifecycle load, not population size
# - Precise equipment deployment (Scanners vs Data Entry)
#
# This insight is derived entirely from relative, aggregated metrics and avoids
# assumptions about individual behavior.
#
# *Note: We applied a volume filter (>1000) to ensure these are significant operational centers, not statistical noise.*
