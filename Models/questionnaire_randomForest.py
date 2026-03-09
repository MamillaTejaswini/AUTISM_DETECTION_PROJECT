#Version1
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Load dataset
# df = pd.read_csv("C:/Users/mamil/OneDrive/Documents/autism_detection_project/Data/Autism-Child-Data.csv")

# # Convert target
# df["Class/ASD"] = df["Class/ASD"].map({"YES": 1, "NO": 0})

# # Drop non-numeric columns if needed
# df = df.drop(columns=["age_desc"], errors="ignore")

# X = df.drop("Class/ASD", axis=1)
# y = df["Class/ASD"]

# # Simple split (NO stratification)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Random Forest
# model = RandomForestClassifier(n_estimators=200, random_state=42)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Initial RF Results")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("F1:", f1_score(y_test, y_pred))

#version2
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# ===============================
# 1. Load Dataset
# ===============================

df = pd.read_csv("../Data/Autism-Child-Data.csv")

# Target
y = df["Class/ASD"].map({"YES": 1, "NO": 0})

# ===============================
# 2. Remove Top 3 Correlated Features
# Removed: A4_Score, A9_Score, A10_Score
# ===============================

num_cols = [
    "A1_Score","A2_Score","A3_Score","A5_Score",
    "A6_Score","A7_Score","A8_Score",
    "age"
]

cat_cols = [
    "jundice",
    "austim",
    "age_desc",
]

X = df[num_cols + cat_cols]

# ===============================
# 3. Preprocessing Pipelines
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

# ===============================
# 4. Full Pipeline (Logistic Regression)
# ===============================

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

# ===============================
# 5. Train-Test Split (Stratified)
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 6. Train Model
# ===============================

pipeline.fit(X_train, y_train)

# ===============================
# 7. Test Evaluation
# ===============================

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n=== Test Set Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ===============================
# 8. 5-Fold Cross Validation
# ===============================

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')

print("\n=== Cross Validation ===")
print("Mean F1:", cv_scores.mean())
print("Std F1:", cv_scores.std())