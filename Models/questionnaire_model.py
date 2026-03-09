import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
# ===============================
# 1. Load Dataset
# ===============================

df = pd.read_csv("../Data/Autism-Child-Data.csv")

# Target
y = df["Class/ASD"].map({"YES": 1, "NO": 0})

# ===============================
# 2. Add 5% Noise to Questionnaire Scores
# ===============================

np.random.seed(42)

question_cols = [
    "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score",
    "A6_Score","A7_Score","A8_Score","A9_Score","A10_Score"
]

df_noisy = df.copy()

n_rows = df.shape[0]
n_noise = int(0.05 * n_rows)  # 5% rows

for col in question_cols:
    noise_indices = np.random.choice(n_rows, n_noise, replace=False)
    df_noisy.loc[noise_indices, col] = 1 - df_noisy.loc[noise_indices, col]
# ===============================
# 3. Feature Engineering
# ===============================

df_noisy["total_deficit_score"] = df_noisy[question_cols].sum(axis=1)
df_noisy["deficit_ratio"] = df_noisy["total_deficit_score"] / 10
# ===============================
# 4. Feature Setup
# ===============================


cat_cols = [
    "jundice",
    "austim",
    "age_desc",
]
num_cols = question_cols + [
    "age",
    "total_deficit_score",
    "deficit_ratio"
]
X = df_noisy[num_cols + cat_cols]

# ===============================
# 4. Preprocessing
# ===============================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

# ===============================
# 5. Train-Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# ===============================
# 6. Hyperparameter Tuning (Step-2)
# ===============================

param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l1", "l2"],
    "model__solver": ["liblinear"]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:", grid_search.best_params_)
print("Best CV F1:", grid_search.best_score_)

# ===============================
# 7. Evaluation on Test Set
# ===============================

y_prob = best_model.predict_proba(X_test)[:, 1]

# Default threshold (0.5)
y_pred_default = (y_prob >= 0.5).astype(int)

print("\n=== Test Performance (Default 0.5 Threshold) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_default))
print("Precision:", precision_score(y_test, y_pred_default))
print("Recall:", recall_score(y_test, y_pred_default))
print("F1 Score:", f1_score(y_test, y_pred_default))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_default))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# ===============================
# 8. Threshold Optimization (Step-3)
# ===============================

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print("\nOptimal Threshold for Maximum F1:", best_threshold)

y_pred_optimal = (y_prob >= best_threshold).astype(int)

print("\n=== Optimized Threshold Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred_optimal))
print("Precision:", precision_score(y_test, y_pred_optimal))
print("Recall:", recall_score(y_test, y_pred_optimal))
print("F1 Score:", f1_score(y_test, y_pred_optimal))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_optimal))


# ===============================
# 9. Cross Validation on Full Data
# ===============================

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='f1')

print("\n=== Cross Validation After Tuning ===")
print("Mean F1:", cv_scores.mean())
print("Std F1:", cv_scores.std())


