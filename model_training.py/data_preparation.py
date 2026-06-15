"""
Data Preparation Script for Model Training
Steps: Load cleaned data → Identify target → Split into X and y → Train/Test Split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================================
# STEP 1: LOAD THE CLEANED DATASET
# ============================================================================
print("=" * 70)
print("STEP 1: LOADING CLEANED DATASET")
print("=" * 70)

# Load Titanic dataset
df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")

print("\n✓ Dataset loaded successfully!")
print(f"Dataset Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# ============================================================================
# STEP 2: BASIC DATA CLEANING (to ensure clean dataset)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: DATA CLEANING")
print("=" * 70)

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values
numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"✓ Data cleaned. Missing values remaining: {df.isnull().sum().sum()}")

# ============================================================================
# STEP 3: IDENTIFY TARGET VARIABLE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: IDENTIFYING TARGET VARIABLE")
print("=" * 70)

# For Titanic dataset, the target variable is 'Survived'
target_variable = 'Survived'
print(f"\nTarget Variable: '{target_variable}'")
print(f"Target values (unique): {df[target_variable].unique()}")
print(f"Target value counts:\n{df[target_variable].value_counts()}")

# ============================================================================
# STEP 4: SEPARATE FEATURES AND TARGET
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: SEPARATING FEATURES AND TARGET")
print("=" * 70)

# Define features (X) - all columns except the target
X = df.drop(columns=[target_variable])

# Define target (y)
y = df[target_variable]

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nFeature columns ({len(X.columns)} total):")
print(list(X.columns))

# ============================================================================
# STEP 5: HANDLE CATEGORICAL VARIABLES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

# Identify categorical columns in X
X_categorical = X.select_dtypes(include=['object']).columns
print(f"\nCategorical columns in features: {list(X_categorical)}")

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=X_categorical, drop_first=True)
print(f"✓ Features after encoding: {X.shape}")

# ============================================================================
# STEP 6: TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: TRAIN/TEST SPLIT")
print("=" * 70)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Ensures both sets have similar class distribution
)

print(f"\nTrain set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

print(f"\nTraining features (X_train) shape: {X_train.shape}")
print(f"Training target (y_train) shape: {y_train.shape}")
print(f"Testing features (X_test) shape: {X_test.shape}")
print(f"Testing target (y_test) shape: {y_test.shape}")

# Display class distribution in both sets
print(f"\n--- Training Set Class Distribution ---")
print(y_train.value_counts())
print(f"\n--- Testing Set Class Distribution ---")
print(y_test.value_counts())

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETE")
print("=" * 70)
print(f"""
Summary:
- Original dataset: {df.shape[0]} rows × {df.shape[1]} columns
- Target variable: '{target_variable}'
- Features (X): {X.shape[1]} columns (after encoding)
- Training set: {X_train.shape[0]} samples
- Testing set: {X_test.shape[0]} samples
- Random state: 42 (fixed for reproducibility)

Ready for model training!
""")
