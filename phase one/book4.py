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
# # HeatMap baby

# %%
# MODIFICATION: Import processed data from book1
# This fixes the "Absolute Path" crash and ensures consistency.
from book1 import district_df

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 227
sns.set_style("whitegrid")

# %% [markdown]
# ## Re-Aggregate to District Level
# *Since book1 provides Pincode granularity, we roll it up to District level for State comparison.*

# %%
# 1. Group Pincodes back into Districts
district_agg = district_df.groupby(['state', 'district'])[[
    'total_enrolments', 'demo_activity', 'bio_activity'
]].sum().reset_index()

# 2. Recalculate Metrics for Visualization
district_agg["total_update_activity"] = (
    district_agg["demo_activity"] + district_agg["bio_activity"]
)

district_agg["update_to_enrolment_ratio"] = (
    district_agg["total_update_activity"] /
    district_agg["total_enrolments"].replace(0, np.nan)
)

# %% [markdown]
# ## Visualization: Regional Pressure Distribution

# %%
sns.set_context("talk")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

# 1. CLEAN DATA
plot_data = district_agg.copy()

# Fix District Names
if 'district' in plot_data.columns:
    plot_data['district'] = plot_data['district'].astype(str).str.replace('?', '-')

# Fix State Names
plot_data['state'] = plot_data['state'].str.title().str.strip()
plot_data['state'] = plot_data['state'].replace({
    'Westbengal': 'West Bengal',
    'Daman And Diu': 'Daman & Diu', 
    'Dadra And Nagar Haveli': 'Dadra & Nagar Haveli',
    'Andaman And Nicobar Islands': 'A & N Islands'
})

# 2. FILTER OUTLIERS & SELECT TOP STATES
# Identify Extreme Outliers (> 150 ratio)
outliers = plot_data[plot_data["update_to_enrolment_ratio"] > 150].sort_values("update_to_enrolment_ratio", ascending=False)
normal_data = plot_data[plot_data["update_to_enrolment_ratio"] <= 150]

# Filter: Only show Top 20 States with the highest 'Max' pressure
# This keeps the chart readable
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
# Only add if outliers exist to prevent errors
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

# %%