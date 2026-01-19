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
# ## Visualization 1: Spatial Distribution (Update Pressure)
# *Replaces the simple top-10 list with a distribution heatmap across states.*

# %%
sns.set_context("talk")
sns.set_style("whitegrid") # Grid is essential for reading the strip plot values
plt.rcParams["figure.figsize"] = (14, 10)

# 1. PREPARE DATA
plot_data = region_df.copy()

# 2. FILTER OUTLIERS & SELECT TOP STATES
# Identify Extreme Outliers (> 150 ratio) to keep the chart readable
outliers = plot_data[plot_data["update_to_enrolment_ratio"] > 150].sort_values("update_to_enrolment_ratio", ascending=False)
normal_data = plot_data[plot_data["update_to_enrolment_ratio"] <= 150]

# Filter: Only show Top 20 States with the highest 'Max' pressure to reduce clutter
top_states_list = normal_data.groupby('state')['update_to_enrolment_ratio'].max().sort_values(ascending=False).head(20).index
filtered_data = normal_data[normal_data['state'].isin(top_states_list)]

# 3. PLOT: HORIZONTAL STRIP PLOT
ax = sns.stripplot(
    data=filtered_data,
    y="state",
    x="update_to_enrolment_ratio",
    hue="update_to_enrolment_ratio",
    palette="rocket_r",
    size=7,
    alpha=0.7,
    jitter=0.25,
    edgecolor="#555555",
    linewidth=0.5,
    order=top_states_list
)

# Vertical Gridlines for Readability
ax.grid(True, axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

# 4. TITLES
plt.figtext(0.5, 0.96, "Spatial Distribution: Aadhaar Update Pressure (Top 20 States)", 
            fontsize=24, weight='bold', ha='center')

plt.figtext(0.5, 0.92, "Focusing on districts with ratio < 150. Extreme outliers excluded from visual.", 
            fontsize=14, color='#666666', ha='center')

# 5. CUSTOMIZE AXES
plt.ylabel("")
plt.xlabel("Update-to-Enrolment Ratio (Updates per New Enrolment)")

# Remove Legend
if ax.legend_:
    ax.legend_.remove()

# 6. ADD "OUTLIER BOX"
if not outliers.empty:
    outlier_text = "!! EXTREME OUTLIERS (OFF-CHART) !!:\n" + "\n".join(
        [f"• {row['district']} ({row['state']}): {row['update_to_enrolment_ratio']:.0f}" 
         for _, row in outliers.head(5).iterrows()]
    )

    plt.text(
        x=0.98, y=0.02, # Bottom Right position
        s=outlier_text,
        transform=ax.transAxes,
        fontsize=12,
        color="#800000",
        bbox=dict(boxstyle="round,pad=0.5", fc="#ffeaea", ec="#800000", alpha=0.9),
        ha="right",
        va="bottom"
    )

# 7. LAYOUT
sns.despine(left=True, bottom=True)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()

# %% [markdown]
# ## Insight: Update-Heavy but Enrolment-Light Regions
#
# ### The strip plot reveals that while most districts maintain a healthy balance, specific "cluster states" exhibit disproportionately high update activity.
#
# This indicates that operational load in these regions (especially the red/orange outliers) is driven primarily by identity maintenance rather than new enrolments.
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

# --- HANDLE LONG NAMES ---
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

# Reset style to white for bar chart (looks cleaner than grid)
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

# --- CREATE SPACE FOR LEGEND ---
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

# --- WRAP LEGEND TEXT ---
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
