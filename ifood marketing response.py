# ============================================================
# iFood Marketing Campaign Response - Random Forest Prediction
#  - Light feature engineering
#  - Balanced class weights (no SMOTE)
#  - GridSearchCV with tuned max_features
#  - Cost-sensitive threshold optimization
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')

# ----------------- Settings ----------------- #

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

# Business cost assumptions (adjust based on campaign economics)
COST_FN = 5000   # cost of missing a potential responder (false negative)
COST_FP = 300    # cost of contacting a non-responder (false positive)

# ----------------- Load Data ----------------- #

os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    df = pd.read_csv('ifood_df.csv')
except FileNotFoundError:
    print("Error: 'ifood_df.csv' not found.")
    raise SystemExit

print("iFood Marketing Campaign Response - Random Forest Model\n")
print("Data preview:")
print(df.head())
print(f"\nShape: {df.shape}")

# ----------------- Target & Basic Cleaning ----------------- #

target_col = 'Response'  # 0 = no response, 1 = response

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found!")

print(f"\nTarget column: {target_col}")
print(f"Class distribution:\n{df[target_col].value_counts()}")
response_rate = (df[target_col] == 1).mean() * 100
print(f"Response rate: {response_rate:.2f}%\n")

# Separate features and target
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# Drop leakage columns (summary of campaigns)
for col in ['AcceptedCmpOverall']:
    if col in X.columns:
        print(f"Dropping leakage column: {col}")
        X = X.drop(columns=[col])

# Drop constant columns (single-valued features)
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    print(f"Dropping constant columns: {constant_cols}\n")
    X = X.drop(columns=constant_cols)

# ----------------- Feature Engineering ----------------- #

print("Creating engineered features...")

# Total children in household
if {'Kidhome', 'Teenhome'}.issubset(X.columns):
    X['TotalChildren'] = X['Kidhome'] + X['Teenhome']

# Total purchases across channels
if {
    'NumDealsPurchases', 'NumWebPurchases',
    'NumCatalogPurchases', 'NumStorePurchases'
}.issubset(X.columns):
    X['TotalPurchases'] = (
        X['NumDealsPurchases'] + X['NumWebPurchases'] +
        X['NumCatalogPurchases'] + X['NumStorePurchases']
    )

# Total spending across product categories
if {
    'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
}.issubset(X.columns) and 'MntTotal' not in X.columns:
    X['MntTotal'] = (
        X['MntWines'] + X['MntFruits'] + X['MntMeatProducts'] +
        X['MntFishProducts'] + X['MntSweetProducts'] + X['MntGoldProds']
    )

print(f"Total features after engineering: {len(X.columns)}\n")

# ----------------- Encode & Handle Missing Values ----------------- #

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print("Feature summary:")
print(f"  Categorical: {len(cat_cols)}")
print(f"  Numerical:   {len(num_cols)}")

# Label encode categorical features
encoders = {}
X_enc = X.copy()
for col in cat_cols:
    le = LabelEncoder()
    X_enc[col] = le.fit_transform(X_enc[col].astype(str))
    encoders[col] = le

# Fill missing values with median
if X_enc.isnull().sum().sum() > 0:
    for col in X_enc.columns:
        if X_enc[col].isnull().sum() > 0:
            X_enc[col] = X_enc[col].fillna(X_enc[col].median())

# ----------------- Train / Test Split ----------------- #

X_train_enc, X_test_enc, y_train, y_test = train_test_split(
    X_enc, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nData split:")
print(f"  Training: {len(X_train_enc)} samples")
print(f"  Test:     {len(X_test_enc)} samples\n")

# ----------------- Random Forest with GridSearchCV ----------------- #

print("Training Random Forest with GridSearchCV...\n")

base_rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'  # handle class imbalance internally
)

# Hyperparameter grid (32 combinations)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']   # decorrelate trees, reduce overfitting
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

grid = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_enc, y_train)
best_model = grid.best_estimator_

print("\nBest hyperparameters:")
for param, value in grid.best_params_.items():
    print(f"  {param}: {value}")

# ----------------- Evaluation at Default Threshold (0.5) ----------------- #

y_proba = best_model.predict_proba(X_test_enc)[:, 1]
y_pred_default = (y_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred_default)
roc = roc_auc_score(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
cv_score = grid.best_score_
cm_default = confusion_matrix(y_test, y_pred_default)

print(f"\nCross-validation ROC-AUC: {cv_score:.4f}")
print(f"\nTest set performance (threshold 0.50):")
print(f"  Accuracy:          {acc:.4f}")
print(f"  ROC-AUC:           {roc:.4f}")
print(f"  Average Precision: {ap:.4f}")

print(f"\nConfusion matrix (threshold 0.50):")
print(cm_default)

print(f"\nClassification report (threshold 0.50):")
print(
    classification_report(
        y_test,
        y_pred_default,
        zero_division=0,
        target_names=['No Response', 'Response']
    )
)

# ----------------- Cost-Sensitive Threshold Optimization ----------------- #

print("\n" + "=" * 70)
print("Cost-Sensitive Threshold Optimization")
print("=" * 70)

thresholds = np.linspace(0.05, 0.95, 19)
best_cost = None
best_thresh = None
best_stats = None

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    total_cost = fp * COST_FP + fn * COST_FN
    
    if best_cost is None or total_cost < best_cost:
        best_cost = total_cost
        best_thresh = t
        best_stats = (tn, fp, fn, tp)

tn, fp, fn, tp = best_stats
print(f"\nOptimal threshold: {best_thresh:.2f}")
print(f"Minimum expected cost: ${best_cost:,.0f}")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives:  {tp}")

y_pred_opt = (y_proba >= best_thresh).astype(int)
acc_opt = accuracy_score(y_test, y_pred_opt)
print(f"\nAccuracy at optimal threshold: {acc_opt:.4f}\n")

# ----------------- Visualizations ----------------- #

print("Generating visualizations...\n")

output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# 1. Feature importance
print("Creating Figure 1: Feature Importance...")
feature_importance = pd.DataFrame({
    'feature': X_enc.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

top20 = feature_importance.head(20)

plt.figure(figsize=(12, 8))
plt.barh(range(len(top20)), top20['importance'], color='steelblue', alpha=0.8)
plt.yticks(range(len(top20)), top20['feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 20 Feature Importances - Random Forest', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: feature_importance.png")

# 2. Confusion matrix (default threshold)
print("Creating Figure 2: Confusion Matrix (0.50)...")
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_default,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=True,
    xticklabels=['No Response', 'Response'],
    yticklabels=['No Response', 'Response']
)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix - Threshold 0.50', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_050.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: confusion_matrix_050.png")

# 3. Confusion matrix (cost-optimal threshold)
print("Creating Figure 3: Confusion Matrix (cost-optimal)...")
cm_opt = confusion_matrix(y_test, y_pred_opt)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_opt,
    annot=True,
    fmt='d',
    cmap='Oranges',
    cbar=True,
    xticklabels=['No Response', 'Response'],
    yticklabels=['No Response', 'Response']
)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title(f'Confusion Matrix - Threshold {best_thresh:.2f} (Cost-Optimal)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_cost_optimal.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: confusion_matrix_cost_optimal.png")

# 4. ROC curve
print("Creating Figure 4: ROC Curve...")
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Random Forest', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: roc_curve.png")

# 5. Precision-Recall curve
print("Creating Figure 5: Precision-Recall Curve...")
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve - Random Forest', fontsize=14, fontweight='bold')
plt.legend(loc="upper right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: precision_recall_curve.png")

# 6. Performance metrics bar chart
print("Creating Figure 6: Performance Metrics...")
metrics = {
    'Accuracy (0.50)': acc,
    'ROC-AUC': roc,
    'Avg Precision': ap,
    'CV ROC-AUC': cv_score
}

plt.figure(figsize=(10, 6))
bars = plt.bar(
    range(len(metrics)),
    list(metrics.values()),
    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
    alpha=0.8,
    edgecolor='black'
)
plt.xticks(range(len(metrics)), list(metrics.keys()), fontsize=11)
plt.ylabel('Score', fontsize=12)
plt.title('Random Forest Performance Metrics', fontsize=14, fontweight='bold')
plt.ylim([0, 1])

for bar, value in zip(bars, metrics.values()):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f'{value:.3f}',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: performance_metrics.png")

# 7. Feature analysis by response
print("Creating Figure 7: Feature Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if 'Age' in df.columns:
    axes[0, 0].hist(
        [df[df[target_col] == 0]['Age'], df[df[target_col] == 1]['Age']],
        bins=15,
        label=['No Response', 'Response'],
        alpha=0.7,
        color=['red', 'green']
    )
    axes[0, 0].set_xlabel('Age', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Age Distribution by Response', fontsize=11, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

if 'Income' in df.columns:
    axes[0, 1].hist(
        [df[df[target_col] == 0]['Income'], df[df[target_col] == 1]['Income']],
        bins=15,
        label=['No Response', 'Response'],
        alpha=0.7,
        color=['red', 'green']
    )
    axes[0, 1].set_xlabel('Income', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].set_title('Income Distribution by Response', fontsize=11, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

if {'MntTotal', 'Recency'}.issubset(df.columns):
    axes[1, 0].scatter(
        df['Recency'],
        df['MntTotal'],
        c=df[target_col],
        cmap='bwr',
        alpha=0.6
    )
    axes[1, 0].set_xlabel('Recency (days since last purchase)', fontsize=10)
    axes[1, 0].set_ylabel('Total Amount Spent', fontsize=10)
    axes[1, 0].set_title('Spending vs Recency by Response', fontsize=11, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

if {'NumWebPurchases', 'NumCatalogPurchases'}.issubset(df.columns):
    axes[1, 1].scatter(
        df['NumWebPurchases'],
        df['NumCatalogPurchases'],
        c=df[target_col],
        cmap='bwr',
        alpha=0.6
    )
    axes[1, 1].set_xlabel('NumWebPurchases', fontsize=10)
    axes[1, 1].set_ylabel('NumCatalogPurchases', fontsize=10)
    axes[1, 1].set_title('Web vs Catalog Purchases by Response', fontsize=11, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: feature_analysis.png")

# ----------------- Save Summary Report ----------------- #

print("\nSaving summary report...")

summary_path = os.path.join(output_dir, 'model_summary.txt')
with open(summary_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("iFood Marketing Campaign Response - Random Forest Model\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("Dataset Overview:\n")
    f.write(f"  Total samples: {len(df)}\n")
    f.write(f"  Training samples: {len(X_train_enc)}\n")
    f.write(f"  Test samples: {len(X_test_enc)}\n")
    f.write(f"  Number of features: {len(X_enc.columns)}\n")
    f.write(f"  Response rate: {response_rate:.2f}%\n\n")
    
    f.write("Best Hyperparameters:\n")
    for param, value in grid.best_params_.items():
        f.write(f"  {param}: {value}\n")
    
    f.write("\nPerformance Metrics (threshold 0.50):\n")
    f.write(f"  Cross-validation ROC-AUC: {cv_score:.4f}\n")
    f.write(f"  Test Accuracy:            {acc:.4f}\n")
    f.write(f"  Test ROC-AUC:             {roc:.4f}\n")
    f.write(f"  Average Precision:        {ap:.4f}\n\n")
    
    f.write("Cost-Sensitive Optimization:\n")
    f.write(f"  Optimal Threshold:        {best_thresh:.2f}\n")
    f.write(f"  Minimum Expected Cost:    ${best_cost:,.0f}\n")
    f.write(f"  Confusion Matrix (TN, FP, FN, TP): {tn}, {fp}, {fn}, {tp}\n\n")
    
    f.write("Top 10 Most Important Features:\n")
    for _, row in feature_importance.head(10).iterrows():
        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

print(f"  ✓ Saved: model_summary.txt")

print("\n" + "=" * 70)
print("Analysis complete!")
print("=" * 70)
print(f"\nGenerated files in '{output_dir}/' directory:")
print("  1. feature_importance.png")
print("  2. confusion_matrix_050.png")
print("  3. confusion_matrix_cost_optimal.png")
print("  4. roc_curve.png")
print("  5. precision_recall_curve.png")
print("  6. performance_metrics.png")
print("  7. feature_analysis.png")
print("  8. model_summary.txt")
print()
