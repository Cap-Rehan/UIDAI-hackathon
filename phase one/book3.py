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


plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 150
sns.set_style("whitegrid")

# %% [markdown]
# ## Load Datasets (Using Relative Paths)

# %%
# FIX 1: Using relative paths instead of absolute paths
# This ensures code works on any machine, not just the original author's

DEMO1_PATH = 'api_data_aadhar_demographic/api_data_aadhar_demographic_0_500000.csv'
DEMO2_PATH = 'api_data_aadhar_demographic/api_data_aadhar_demographic_500000_1000000.csv'
DEMO3_PATH = 'api_data_aadhar_demographic/api_data_aadhar_demographic_1000000_1500000.csv'
DEMO4_PATH = 'api_data_aadhar_demographic/api_data_aadhar_demographic_1500000_2000000.csv'
DEMO5_PATH = 'api_data_aadhar_demographic/api_data_aadhar_demographic_2000000_2071700.csv'

BIO1_PATH = 'api_data_aadhar_biometric/api_data_aadhar_biometric_0_500000.csv'
BIO2_PATH = 'api_data_aadhar_biometric/api_data_aadhar_biometric_500000_1000000.csv'
BIO3_PATH = 'api_data_aadhar_biometric/api_data_aadhar_biometric_1000000_1500000.csv'
BIO4_PATH = 'api_data_aadhar_biometric/api_data_aadhar_biometric_1500000_1861108.csv'

# %%
demo1 = pd.read_csv(DEMO1_PATH)
demo2 = pd.read_csv(DEMO2_PATH)
demo3 = pd.read_csv(DEMO3_PATH)
demo4 = pd.read_csv(DEMO4_PATH)
demo5 = pd.read_csv(DEMO5_PATH)

bio1 = pd.read_csv(BIO1_PATH)
bio2 = pd.read_csv(BIO2_PATH)
bio3 = pd.read_csv(BIO3_PATH)
bio4 = pd.read_csv(BIO4_PATH)

# %%
demo = pd.concat([demo1, demo2, demo3, demo4, demo5], ignore_index=True)
bio = pd.concat([bio1, bio2, bio3, bio4], ignore_index=True)

print(f"Demo records: {len(demo)}")
print(f"Bio records: {len(bio)}")

# %%
print("Demo columns:", demo.columns.tolist())
print("Bio columns:", bio.columns.tolist())

# %% [markdown]
# ## Feature Engineering: Age-Specific Activity

# %%
# Keep age groups SEPARATE (unlike book1 which combines them)
# This is essential for age-driven analysis

demo["activity_5_17"] = demo["demo_age_5_17"]
demo["activity_17_plus"] = demo["demo_age_17_"]

bio["activity_5_17"] = bio["bio_age_5_17"]
bio["activity_17_plus"] = bio["bio_age_17_"]

# %% [markdown]
# ## Aggregate at Pincode Level (FIX 3: Zoom Level)

# %%
# FIX 3: Using pincode level instead of district level
# This matches book1 and book2 for consistency

demo_pin = (
    demo.groupby(["state", "district", "pincode"], as_index=False)[
        ["activity_5_17", "activity_17_plus"]
    ].sum()
)

bio_pin = (
    bio.groupby(["state", "district", "pincode"], as_index=False)[
        ["activity_5_17", "activity_17_plus"]
    ].sum()
)

# %%
print(f"Demo pincodes: {len(demo_pin)}")
print(f"Bio pincodes: {len(bio_pin)}")

# %% [markdown]
# ## Merge Datasets

# %%
pincode_df = (
    demo_pin
    .merge(bio_pin, on=["state", "district", "pincode"], suffixes=("_demo", "_bio"), how="outer")
)

pincode_df.fillna(0, inplace=True)

# Combine demo and bio activity by age group
pincode_df["activity_5_17"] = (
    pincode_df["activity_5_17_demo"] +
    pincode_df["activity_5_17_bio"]
)

pincode_df["activity_17_plus"] = (
    pincode_df["activity_17_plus_demo"] +
    pincode_df["activity_17_plus_bio"]
)

# Keep only needed columns
pincode_df = pincode_df[
    ["state", "district", "pincode", "activity_5_17", "activity_17_plus"]
]

# %%
print(f"Total pincodes: {len(pincode_df)}")
pincode_df.head()

# %% [markdown]
# ## Core Metric: Age-Skew Ratio

# %%
pincode_df["total_update_activity"] = (
    pincode_df["activity_5_17"] +
    pincode_df["activity_17_plus"]
)

# Adult share (17+)
pincode_df["age_17_plus_share"] = (
    pincode_df["activity_17_plus"] /
    pincode_df["total_update_activity"].replace(0, np.nan)
)

# Child share (5-17) - FIX 2: Now we calculate BOTH
pincode_df["age_5_17_share"] = (
    pincode_df["activity_5_17"] /
    pincode_df["total_update_activity"].replace(0, np.nan)
)

# %%
pincode_df[["age_17_plus_share", "age_5_17_share"]].describe()

# %% [markdown]
# ## Apply Volume Filter (Remove Statistical Noise)

# %%
# Only analyze pincodes with significant activity
VOLUME_THRESHOLD = 100
filtered = pincode_df[
    pincode_df["total_update_activity"] >= VOLUME_THRESHOLD
].copy()

print(f"Pincodes after filtering: {len(filtered)}")

# %%
# Calculate national median for reference
national_median_adult = filtered["age_17_plus_share"].median()
national_median_child = filtered["age_5_17_share"].median()

print(f"National Median - Adult (17+) Share: {national_median_adult:.2%}")
print(f"National Median - Child (5-17) Share: {national_median_child:.2%}")

# %% [markdown]
# ## Identify Adult-Heavy Pincodes (Permanent Centers Needed)

# %%
# Top 10% by adult share
adult_threshold = filtered["age_17_plus_share"].quantile(0.90)

adult_heavy = (
    filtered[filtered["age_17_plus_share"] >= adult_threshold]
    .sort_values("age_17_plus_share", ascending=False)
)

print(f"Adult-heavy pincodes found: {len(adult_heavy)}")
adult_heavy.head(10)

# %% [markdown]
# ## Identify Child-Heavy Pincodes (School Camps Needed)

# %%
# FIX 2: Now we look at the OTHER end - high child share
# Top 10% by child share
child_threshold = filtered["age_5_17_share"].quantile(0.90)

child_heavy = (
    filtered[filtered["age_5_17_share"] >= child_threshold]
    .sort_values("age_5_17_share", ascending=False)
)

print(f"Child-heavy pincodes found: {len(child_heavy)}")
child_heavy.head(10)

# %% [markdown]
# ## Visualization 1: Adult-Heavy Pincodes (Permanent Centers)

# %%
# Prepare data
top10_adult = adult_heavy.head(10).copy()
top10_adult['pincode'] = top10_adult['pincode'].astype(str)

# Create label combining district and pincode for clarity
top10_adult['location'] = top10_adult['district'] + " (" + top10_adult['pincode'] + ")"

# %%
# Create figure
fig1, ax1 = plt.subplots(figsize=(12, 8), dpi=150)

sns.set_context("talk")
sns.set_style("white")

# Plot
sns.barplot(
    data=top10_adult,
    y="location",
    x="age_17_plus_share",
    hue="state",
    palette="crest",
    dodge=False,
    ax=ax1
)

# Median reference line
ax1.axvline(
    national_median_adult,
    color="#FF4B4B",
    linestyle="--",
    linewidth=2,
    alpha=0.8
)

ax1.text(
    x=national_median_adult + 0.01,
    y=-0.7,
    s=f"National Median: {national_median_adult:.2f}",
    color="#FF4B4B",
    weight="bold",
    ha="left",
    va="center"
)

# Titles
ax1.set_title("Top 10 Pincodes: Adult (17+) Update Concentration\n", 
              fontsize=20, weight='bold', loc='center')
ax1.text(0.5, 1.02, "Recommendation: Deploy Permanent Aadhaar Centers", 
         fontsize=12, color='#2ecc71', weight='bold', ha='center', transform=ax1.transAxes)

# Data labels
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.2f', padding=-40, fontsize=11, color='white', weight='bold')

# Cleanup
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.set_xticks([])
ax1.set_xlim(0, 1.15)
sns.despine(left=True, bottom=True)

# Legend
ax1.legend(loc="lower right", frameon=False, title="")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Insight: Adult-Heavy Pincodes
#
# ### These pincodes show 80-90%+ of update activity coming from adults (17+).
#
# **Why this matters:**
# - Adults visit Aadhaar centers individually (not in groups)
# - They come during working hours or need evening/weekend access
# - Updates are for address changes, mobile linking, biometric refresh
#
# **Recommendation: Permanent Aadhaar Centers**
# - Fixed location with regular staff
# - Extended operating hours (evenings/weekends)
# - Walk-in service model

# %% [markdown]
# ## Visualization 2: Child-Heavy Pincodes (School Camps)

# %%
# Prepare data
top10_child = child_heavy.head(10).copy()
top10_child['pincode'] = top10_child['pincode'].astype(str)

# Create label combining district and pincode for clarity
top10_child['location'] = top10_child['district'] + " (" + top10_child['pincode'] + ")"

# %%
# Create figure
fig2, ax2 = plt.subplots(figsize=(12, 8), dpi=150)

sns.set_context("talk")
sns.set_style("white")

# Plot - Using warm colors (orange/yellow) for children
sns.barplot(
    data=top10_child,
    y="location",
    x="age_5_17_share",
    hue="state",
    palette="YlOrRd",
    dodge=False,
    ax=ax2
)

# Median reference line
ax2.axvline(
    national_median_child,
    color="#3498db",
    linestyle="--",
    linewidth=2,
    alpha=0.8
)

ax2.text(
    x=national_median_child + 0.01,
    y=-0.7,
    s=f"National Median: {national_median_child:.2f}",
    color="#3498db",
    weight="bold",
    ha="left",
    va="center"
)

# Titles
ax2.set_title("Top 10 Pincodes: Child (5-17) Update Concentration\n", 
              fontsize=20, weight='bold', loc='center')
ax2.text(0.5, 1.02, "Recommendation: Deploy School-Based Aadhaar Camps", 
         fontsize=12, color='#e67e22', weight='bold', ha='center', transform=ax2.transAxes)

# Data labels
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.2f', padding=-40, fontsize=11, color='white', weight='bold')

# Cleanup
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.set_xticks([])
ax2.set_xlim(0, 0.8)
sns.despine(left=True, bottom=True)

# Legend
ax2.legend(loc="lower right", frameon=False, title="")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Insight: Child-Heavy Pincodes
#
# ### These pincodes show 40-60%+ of update activity coming from children (5-17).
#
# **Why this matters:**
# - Children require MANDATORY biometric updates at ages 5 and 15
# - Children are already grouped in schools (efficient deployment)
# - Parents/guardians must accompany (complex logistics if done individually)
#
# **Recommendation: School-Based Aadhaar Camps**
# - Partner with schools in these pincodes
# - Conduct bulk enrollment/update drives
# - Schedule during school hours (efficient, captive audience)
# - One camp can process hundreds of updates in a day

# %% [markdown]
# ---
# ## Comparative View: Deployment Strategy by Pincode

# %%
# Create a combined view showing both extremes

# Get top 5 from each category
top5_adult = adult_heavy.head(5).copy()
top5_adult['category'] = 'Adult Heavy → Permanent Center'
top5_adult['share'] = top5_adult['age_17_plus_share']

top5_child = child_heavy.head(5).copy()
top5_child['category'] = 'Child Heavy → School Camp'
top5_child['share'] = top5_child['age_5_17_share']

# Combine
combined = pd.concat([top5_adult, top5_child], ignore_index=True)
combined['pincode'] = combined['pincode'].astype(str)
combined['location'] = combined['district'] + " (" + combined['pincode'] + ")"

# %%
# Create figure
fig3, ax3 = plt.subplots(figsize=(12, 10), dpi=150)

sns.set_context("talk")
sns.set_style("white")

# Plot
sns.barplot(
    data=combined,
    y="location",
    x="share",
    hue="category",
    palette={"Adult Heavy → Permanent Center": "#3498db", "Child Heavy → School Camp": "#e67e22"},
    dodge=False,
    ax=ax3
)

# Titles
ax3.set_title("Age-Based Deployment Strategy: Two Distinct Needs\n", 
              fontsize=20, weight='bold', loc='center')
ax3.text(0.5, 1.02, "Top 5 pincodes from each category with recommended intervention", 
         fontsize=12, color='#666666', ha='center', transform=ax3.transAxes)

# Cleanup
ax3.set_xlabel("Age Group Share of Total Activity")
ax3.set_ylabel("")
sns.despine(left=True, bottom=True)

# Legend
ax3.legend(loc="lower right", frameon=False, title="")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Combined Insight: Age-Driven Service Pressure
#
# ### The same district can have pincodes with completely opposite needs.
#
# By analyzing at the **pincode level** instead of district level, we uncover two distinct service patterns:
#
# | Category | Age Profile | Recommended Deployment |
# |----------|-------------|----------------------|
# | **Adult Heavy** | 80-90% activity from 17+ | Permanent Centers with flexible hours |
# | **Child Heavy** | 40-60% activity from 5-17 | School-based Camps during school hours |
#
# **Why this matters for UIDAI:**
#
# 1. **Resource Efficiency:** Don't deploy permanent centers where school camps would be more effective
# 2. **Higher Throughput:** School camps can process 100s of children in one day
# 3. **Better Compliance:** Mandatory child updates (ages 5 & 15) are captured efficiently
# 4. **Cost Savings:** Camps are temporary; centers are permanent overhead
#
# **Key Finding:** A one-size-fits-all approach wastes resources. Pincode-level age analysis enables precision deployment.
#
# *All findings are based on aggregated activity patterns and do not rely on individual-level assumptions.*

# %% [markdown]
# ---

# %% [markdown]
# ## Data Export: Deployment Recommendations

# %%
# Create actionable export for UIDAI field teams

# Adult-heavy pincodes with recommendations
adult_export = adult_heavy[['state', 'district', 'pincode', 'total_update_activity', 'age_17_plus_share']].head(20).copy()
adult_export['recommendation'] = 'Permanent Aadhaar Center'
adult_export['priority'] = range(1, len(adult_export) + 1)

# Child-heavy pincodes with recommendations
child_export = child_heavy[['state', 'district', 'pincode', 'total_update_activity', 'age_5_17_share']].head(20).copy()
child_export['recommendation'] = 'School-Based Aadhaar Camp'
child_export['priority'] = range(1, len(child_export) + 1)

# %%
print("=== TOP 20: PERMANENT CENTER RECOMMENDATIONS ===")
print(adult_export.to_string(index=False))

# %%
print("\n=== TOP 20: SCHOOL CAMP RECOMMENDATIONS ===")
print(child_export.to_string(index=False))