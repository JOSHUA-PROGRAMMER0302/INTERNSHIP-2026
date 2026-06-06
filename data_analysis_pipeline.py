"""
Comprehensive Data Analysis Pipeline
Steps: Load → Clean → Detect Outliers → Visualize → Dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# ============================================================================
# STEP 1: PROJECT SETUP & LOADING DATA
# ============================================================================
print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

# Load Titanic dataset
df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")

print("\n✓ Dataset loaded successfully!\n")
print("First 5 rows:")
print(df.head())

print(f"\nDataset Shape: {df.shape} (rows, columns)")

print("\nColumn Data Types:")
print(df.dtypes)

print("\nMissing Values Summary:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])  # Only show columns with missing values
print(f"Total missing values: {df.isnull().sum().sum()}")

# ============================================================================
# STEP 2: DATA CLEANING (MISSING VALUES & DUPLICATES)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: DATA CLEANING")
print("=" * 70)

# Track duplicates
duplicates_before = df.duplicated().sum()
print(f"\nDuplicates found: {duplicates_before}")

# Remove duplicates
df = df.drop_duplicates()
duplicates_removed = duplicates_before - df.duplicated().sum()
print(f"Duplicates removed: {duplicates_removed}")

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print(f"\nNumerical columns: {list(numerical_cols)}")
print(f"Categorical columns: {list(categorical_cols)}")

# Impute numerical columns with mean
print("\nImputing numerical columns with mean...")
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
        print(f"  - {col}: filled with mean = {mean_val:.2f}")

# Impute categorical columns with mode
print("\nImputing categorical columns with mode...")
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  - {col}: filled with mode = {mode_val}")

# Verify no missing values remain
missing_after = df.isnull().sum().sum()
print(f"\n✓ Missing values after cleaning: {missing_after}")

# ============================================================================
# STEP 3: OUTLIER DETECTION & REMOVAL (IQR METHOD)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: OUTLIER DETECTION & REMOVAL (IQR Method)")
print("=" * 70)

def remove_outliers_iqr(dataframe, columns=None):
    """
    Remove outliers using Interquartile Range (IQR) method.
    
    Parameters:
    - dataframe: pandas DataFrame
    - columns: list of column names to check (default: all numerical)
    
    Returns:
    - cleaned dataframe, indices of removed rows
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    df_clean = dataframe.copy()
    removed_indices = []
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        removed_indices.extend(df_clean[outlier_mask].index.tolist())
        
        outlier_count = outlier_mask.sum()
        if outlier_count > 0:
            print(f"\n{col}:")
            print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
            print(f"  Outliers detected: {outlier_count}")
    
    # Remove duplicate indices
    removed_indices = list(set(removed_indices))
    df_clean = df_clean.drop(removed_indices)
    
    return df_clean, removed_indices

print(f"\nDataset shape before outlier removal: {df.shape}")
df_clean, removed_idx = remove_outliers_iqr(df, columns=['Age', 'Fare', 'Pclass'])
print(f"\nDataset shape after outlier removal: {df_clean.shape}")
print(f"Total rows removed: {len(removed_idx)}")

df = df_clean

# ============================================================================
# STEP 4: DATA VISUALIZATION (DISTRIBUTION & RELATIONSHIPS)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: DATA VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Histogram + KDE of Age (primary numerical feature)
axes[0].hist(df['Age'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
ax2 = axes[0].twinx()
df['Age'].plot(kind='kde', ax=ax2, color='red', linewidth=2, label='KDE')
axes[0].set_xlabel('Age', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
axes[0].set_title('Distribution of Passenger Age (Histogram + KDE)', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right')

# Plot 2: Correlation Heatmap
numerical_df = df.select_dtypes(include=[np.number])
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, ax=axes[1], cbar_kws={'label': 'Correlation'})
axes[1].set_title('Correlation Heatmap of Numerical Variables', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('visualization_report.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'visualization_report.png'")
plt.show()

# ============================================================================
# STEP 5: DASHBOARD SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: DASHBOARD SUMMARY")
print("=" * 70)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: Boxplot - Fare by Pclass (categorical vs numerical)
ax1 = fig.add_subplot(gs[0, 0])
sns.boxplot(data=df, x='Pclass', y='Fare', palette='Set2', ax=ax1)
ax1.set_title('Fare Distribution by Passenger Class', fontsize=12, fontweight='bold')
ax1.set_xlabel('Passenger Class', fontsize=11)
ax1.set_ylabel('Fare', fontsize=11)

# Plot 2: Boxplot - Age by Survived (categorical vs numerical)
ax2 = fig.add_subplot(gs[0, 1])
sns.boxplot(data=df, x='Survived', y='Age', palette='Set1', ax=ax2)
ax2.set_title('Age Distribution by Survival Status', fontsize=12, fontweight='bold')
ax2.set_xlabel('Survived (0=No, 1=Yes)', fontsize=11)
ax2.set_ylabel('Age', fontsize=11)

# Plot 3: Scatterplot - Top 2 correlated features
# Find top 2 correlated features (excluding self-correlation)
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                          abs(corr_matrix.iloc[i, j])))
corr_pairs.sort(key=lambda x: x[2], reverse=True)
top_feat1, top_feat2, top_corr = corr_pairs[0]

ax3 = fig.add_subplot(gs[1, 0])
scatter = ax3.scatter(df[top_feat1], df[top_feat2], alpha=0.6, c=df['Survived'], 
                     cmap='viridis', s=50, edgecolors='black', linewidth=0.5)
ax3.set_xlabel(f'{top_feat1}', fontsize=11)
ax3.set_ylabel(f'{top_feat2}', fontsize=11)
ax3.set_title(f'Relationship: {top_feat1} vs {top_feat2}\n(Correlation: {top_corr:.3f})', 
             fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax3, label='Survived')

# Plot 4: Distribution comparison - Survived vs Not Survived (Age)
ax4 = fig.add_subplot(gs[1, 1])
df[df['Survived'] == 0]['Age'].hist(bins=20, alpha=0.6, label='Not Survived', 
                                     color='red', ax=ax4, edgecolor='black')
df[df['Survived'] == 1]['Age'].hist(bins=20, alpha=0.6, label='Survived', 
                                     color='green', ax=ax4, edgecolor='black')
ax4.set_xlabel('Age', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Age Distribution by Survival Status', fontsize=12, fontweight='bold')
ax4.legend()

plt.savefig('dashboard_summary.png', dpi=300, bbox_inches='tight')
print("✓ Dashboard saved as 'dashboard_summary.png'")
plt.show()

# ============================================================================
# FINAL SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("FINAL DATA SUMMARY")
print("=" * 70)
print(f"\nFinal dataset shape: {df.shape}")
print(f"Rows removed (duplicates + outliers): {len(removed_idx) + duplicates_removed}")
print(f"\nBasic Statistics:")
print(df.describe())
print(f"\nTop correlated features: {top_feat1} & {top_feat2} (r = {top_corr:.3f})")
print("\n✓ Analysis complete!")
