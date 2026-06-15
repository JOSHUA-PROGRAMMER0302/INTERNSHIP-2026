"""
Comprehensive Exploratory Data Analysis (EDA) Script
Steps: Load Data → Statistical Summary → Correlation Analysis → Distribution Patterns → Structured Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Set style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 12)

# ============================================================================
# STEP 0: LOAD DATA
# ============================================================================
print("=" * 80)
print("STEP 0: LOADING DATA")
print("=" * 80)

# Try to load cleaned_data.csv, fallback to Titanic dataset
try:
    df = pd.read_csv('cleaned_data.csv')
    print("✓ Loaded 'cleaned_data.csv'")
except FileNotFoundError:
    print("⚠ 'cleaned_data.csv' not found. Loading Titanic dataset...")
    df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")
    
    # Quick cleaning
    df = df.drop_duplicates()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    print("✓ Titanic dataset loaded and cleaned")

print(f"\nDataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ============================================================================
# STEP 1: STATISTICAL SUMMARIES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: STATISTICAL SUMMARIES")
print("=" * 80)

# Numerical columns summary
numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print("\n" + "-" * 80)
print("NUMERICAL COLUMNS - STATISTICAL SUMMARY")
print("-" * 80)
print(df[numerical_cols].describe().to_string())

print("\n" + "-" * 80)
print("CATEGORICAL COLUMNS - UNIQUE VALUE COUNTS")
print("-" * 80)

for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())
    print(f"  Unique values: {df[col].nunique()}")

# ============================================================================
# STEP 2: IDENTIFY CORRELATIONS (HEATMAP)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CORRELATION ANALYSIS - GENERATING HEATMAP")
print("=" * 80)

# Calculate correlation matrix for numerical features
correlation_matrix = df[numerical_cols].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.to_string())

# Create figure for heatmap
fig, ax = plt.subplots(figsize=(14, 10))

# Generate heatmap with annotations
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={'label': 'Correlation Coefficient'},
    ax=ax,
    annot_kws={'size': 10, 'weight': 'bold'}
)

ax.set_title('Correlation Matrix Heatmap - Numerical Features', 
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

# Save heatmap
heatmap_path = 'correlation_heatmap.png'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Correlation heatmap saved to: {heatmap_path}")
plt.close()

# ============================================================================
# STEP 3: UNCOVER PATTERNS (DISTRIBUTIONS & PAIRPLOTS)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: UNCOVERING PATTERNS - DISTRIBUTIONS & PAIRPLOTS")
print("=" * 80)

# Identify top 3 numerical features by variance/importance
top_3_features = df[numerical_cols].var().nlargest(3).index.tolist()
print(f"\nTop 3 features by variance: {top_3_features}")

# -------- SUBPLOT 1: HISTOGRAMS WITH KDE FOR TOP 3 FEATURES --------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, feature in enumerate(top_3_features):
    # Histogram with KDE overlay
    axes[idx].hist(df[feature], bins=30, alpha=0.6, color='skyblue', edgecolor='black', density=True)
    
    # KDE overlay
    df[feature].plot(kind='kde', ax=axes[idx], color='red', linewidth=2.5, label='KDE')
    
    # Calculate skewness and kurtosis
    skewness = skew(df[feature].dropna())
    kurt = kurtosis(df[feature].dropna())
    
    axes[idx].set_title(f'{feature} Distribution\n(Skewness: {skewness:.3f}, Kurtosis: {kurt:.3f})',
                        fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(feature, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()

# Save distributions plot
dist_path = 'feature_distributions.png'
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
print(f"✓ Distribution plots saved to: {dist_path}")
plt.close()

# -------- SUBPLOT 2: PAIRPLOT FOR TOP 3 FEATURES --------
print("\nGenerating pairplot for top 3 features...")

# Prepare data for pairplot
pairplot_data = df[top_3_features].copy()

# Find a suitable categorical column for hue (preferably binary target)
hue_col = None
for col in categorical_cols:
    if df[col].nunique() <= 5:  # Use categorical column with ≤5 unique values
        hue_col = col
        break

if hue_col:
    pairplot_data[hue_col] = df[hue_col]
    pairplot = sns.pairplot(pairplot_data, hue=hue_col, diag_kind='kde', 
                             plot_kws={'alpha': 0.6, 's': 50},
                             diag_kws={'linewidth': 2})
    title_text = f'Pairplot of Top 3 Features (colored by {hue_col})'
else:
    pairplot = sns.pairplot(pairplot_data, diag_kind='kde',
                             plot_kws={'alpha': 0.6, 's': 50},
                             diag_kws={'linewidth': 2})
    title_text = 'Pairplot of Top 3 Features'

pairplot.fig.suptitle(title_text, fontsize=16, fontweight='bold', y=1.00)

# Save pairplot
pairplot_path = 'feature_pairplot.png'
plt.savefig(pairplot_path, dpi=300, bbox_inches='tight')
print(f"✓ Pairplot saved to: {pairplot_path}")
plt.close()

# ============================================================================
# STEP 4: STRUCTURED EDA REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: STRUCTURED EDA REPORT")
print("=" * 80)

def extract_top_correlations(corr_matrix, top_n=5):
    """
    Extract top N highest absolute correlations (excluding self-correlations).
    
    Returns:
    - List of tuples: (feature1, feature2, correlation_value)
    """
    # Get upper triangle of correlation matrix (to avoid duplicates)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Stack to get all pairs with correlations
    correlations = upper_triangle.stack().reset_index()
    correlations.columns = ['Feature1', 'Feature2', 'Correlation']
    
    # Sort by absolute correlation and return top N
    correlations['AbsCorrelation'] = correlations['Correlation'].abs()
    correlations = correlations.sort_values('AbsCorrelation', ascending=False)
    
    return correlations.head(top_n)

# Extract top 5 correlations
top_correlations = extract_top_correlations(correlation_matrix, top_n=5)

# Generate report
print("\n" + "█" * 80)
print("█" + " " * 78 + "█")
print("█" + " " * 20 + "COMPREHENSIVE EDA REPORT" + " " * 35 + "█")
print("█" + " " * 78 + "█")
print("█" * 80)

print(f"\n📊 DATASET OVERVIEW")
print("─" * 80)
print(f"  • Total Rows: {df.shape[0]}")
print(f"  • Total Columns: {df.shape[1]}")
print(f"  • Numerical Features: {len(numerical_cols)} {list(numerical_cols)}")
print(f"  • Categorical Features: {len(categorical_cols)} {list(categorical_cols)}")
print(f"  • Missing Values: {df.isnull().sum().sum()}")

print(f"\n📈 NUMERICAL FEATURES STATISTICS")
print("─" * 80)
for col in numerical_cols:
    print(f"\n  {col}:")
    print(f"    Mean: {df[col].mean():.4f}")
    print(f"    Median: {df[col].median():.4f}")
    print(f"    Std Dev: {df[col].std():.4f}")
    print(f"    Range: [{df[col].min():.4f}, {df[col].max():.4f}]")
    print(f"    Skewness: {skew(df[col].dropna()):.4f}")

print(f"\n🔗 TOP 5 FEATURE CORRELATIONS (Influence Factors)")
print("─" * 80)
for idx, row in top_correlations.iterrows():
    print(f"\n  {idx + 1}. {row['Feature1']} ↔ {row['Feature2']}")
    print(f"     Correlation: {row['Correlation']:+.4f} (|{row['AbsCorrelation']:.4f}|)")
    if abs(row['Correlation']) > 0.7:
        relationship = "STRONG positive" if row['Correlation'] > 0 else "STRONG negative"
    elif abs(row['Correlation']) > 0.4:
        relationship = "MODERATE positive" if row['Correlation'] > 0 else "MODERATE negative"
    else:
        relationship = "WEAK positive" if row['Correlation'] > 0 else "WEAK negative"
    print(f"     Relationship: {relationship}")

print(f"\n📁 CATEGORICAL FEATURES DISTRIBUTION")
print("─" * 80)
for col in categorical_cols:
    print(f"\n  {col}:")
    value_counts = df[col].value_counts()
    for val, count in value_counts.items():
        pct = (count / len(df)) * 100
        print(f"    • {val}: {count} ({pct:.1f}%)")

print(f"\n🎯 KEY INSIGHTS & FINDINGS")
print("─" * 80)

# Find highest single correlation
max_corr_row = top_correlations.iloc[0]
print(f"\n  ✓ Strongest Relationship:")
print(f"    {max_corr_row['Feature1']} and {max_corr_row['Feature2']}")
print(f"    show correlation of {max_corr_row['Correlation']:+.4f}")

# Identify highly skewed features
skewed_features = []
for col in numerical_cols:
    skewness = skew(df[col].dropna())
    if abs(skewness) > 1:
        skewed_features.append((col, skewness))

if skewed_features:
    print(f"\n  ⚠ Highly Skewed Features (|skewness| > 1):")
    for feat, skew_val in skewed_features:
        direction = "RIGHT-skewed" if skew_val > 0 else "LEFT-skewed"
        print(f"    • {feat}: {skew_val:.4f} ({direction})")

# Feature importance based on variance
print(f"\n  📊 Features by Variance (Importance):")
variances = df[numerical_cols].var().sort_values(ascending=False)
for feat, var in variances.head(5).items():
    print(f"    • {feat}: {var:.4f}")

print(f"\n{'█' * 80}")
print("█" + " " * 20 + "EDA REPORT COMPLETE" + " " * 40 + "█")
print("█" * 80)

print(f"\n📊 GENERATED VISUALIZATIONS:")
print(f"  ✓ {heatmap_path}")
print(f"  ✓ {dist_path}")
print(f"  ✓ {pairplot_path}")

print(f"\n✅ EDA Analysis Complete! Ready for presentation and reporting.")
print("=" * 80)
