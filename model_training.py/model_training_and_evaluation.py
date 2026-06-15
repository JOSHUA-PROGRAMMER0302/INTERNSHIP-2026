"""
Model Training & Evaluation Script
Steps: Train Random Forest → Evaluate Metrics → Visualize Performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc, roc_auc_score)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# ============================================================================
# STEP 0: LOAD AND PREPARE DATA
# ============================================================================
print("=" * 70)
print("STEP 0: DATA PREPARATION")
print("=" * 70)

# Load Titanic dataset
df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")

# Basic cleaning
df = df.drop_duplicates()

numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Separate features and target
target_variable = 'Survived'
X = df.drop(columns=[target_variable])
y = df[target_variable]

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns, drop_first=True)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Data prepared successfully!")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# ============================================================================
# STEP 1: MODEL TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: MODEL TRAINING - RANDOM FOREST CLASSIFIER")
print("=" * 70)

# Initialize Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1
)

print("\nRandom Forest Configuration:")
print(f"  - Number of trees: 100")
print(f"  - Random state: 42")
print(f"  - Max depth: 15")
print(f"  - Min samples split: 5")

# Fit the model to training data
print("\nFitting model to training data...")
rf_model.fit(X_train, y_train)
print("✓ Model training complete!")

# Make predictions on test data
print("\nMaking predictions on test data...")
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability for positive class
print("✓ Predictions complete!")

# ============================================================================
# STEP 2: ACCURACY & EVALUATION METRICS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: MODEL EVALUATION - ACCURACY & METRICS")
print("=" * 70)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'*' * 70}")
print(f"ACCURACY: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"{'*' * 70}")

# Generate classification report
print("\nCLASSIFICATION REPORT:")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))

# Additional metrics
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nCONFUSION MATRIX:")
print(conf_matrix)

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Feature importance
print("\n" + "-" * 70)
print("TOP 10 MOST IMPORTANT FEATURES:")
print("-" * 70)
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']:.<40} {row['Importance']:.4f}")

# ============================================================================
# STEP 3: VISUALIZING PERFORMANCE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: VISUALIZING PERFORMANCE")
print("=" * 70)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# -------- SUBPLOT 1: CONFUSION MATRIX HEATMAP --------
print("\nGenerating Confusion Matrix Heatmap...")

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Create heatmap
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=True,
    xticklabels=['Did Not Survive', 'Survived'],
    yticklabels=['Did Not Survive', 'Survived'],
    ax=axes[0],
    annot_kws={'size': 14, 'weight': 'bold'},
    cbar_kws={'label': 'Count'}
)

axes[0].set_title('Confusion Matrix - Random Forest Model', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

# Add accuracy annotation
accuracy_text = f'Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)'
axes[0].text(1, -0.15, accuracy_text, ha='center', transform=axes[0].transAxes,
             fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# -------- SUBPLOT 2: ROC CURVE --------
print("Generating ROC Curve with AUC...")

# Calculate ROC curve components
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
axes[1].plot(fpr, tpr, color='#2E86AB', linewidth=2.5, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')

# Plot diagonal reference line (random classifier)
axes[1].plot([0, 1], [0, 1], color='red', linewidth=2, linestyle='--', 
             label='Random Classifier (AUC = 0.5000)')

# Formatting
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
axes[1].set_title('ROC Curve - Random Forest Model', fontsize=14, fontweight='bold')
axes[1].legend(loc='lower right', fontsize=11, framealpha=0.95)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_path = 'model_performance_visualization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")

# Display plot
plt.show()

# ============================================================================
# STEP 4: SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY REPORT")
print("=" * 70)

print(f"""
Model Performance Summary:
{'─' * 70}
Algorithm:              Random Forest Classifier
Number of Trees:        100
Test Set Size:          {X_test.shape[0]} samples

Performance Metrics:
  • Accuracy:           {accuracy:.4f} ({accuracy * 100:.2f}%)
  • ROC-AUC Score:      {roc_auc:.4f}
  
Confusion Matrix Analysis:
  • True Negatives:     {conf_matrix[0, 0]}
  • False Positives:    {conf_matrix[0, 1]}
  • False Negatives:    {conf_matrix[1, 0]}
  • True Positives:     {conf_matrix[1, 1]}

Classification Report Details:
  • Classes analyzed: 2 (Did Not Survive, Survived)
  • See detailed metrics printed above
  
Visualizations Created:
  ✓ Confusion Matrix Heatmap
  ✓ ROC Curve with AUC Score
  
Files Generated:
  ✓ {output_path} (ready for project report)

{'─' * 70}
Model training and evaluation complete!
Ready for project submission.
""")

print("=" * 70)
