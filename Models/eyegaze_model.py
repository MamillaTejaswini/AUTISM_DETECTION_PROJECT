import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
# ---------------------------------
# Function to extract features
# ---------------------------------

def extract_features(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if len(df) < 2:
        return None

    x = df["x"].values
    y = df["y"].values

    duration = df["duration"].values

    # 1. Total fixations
    total_fixations = len(df)

    # 2. Mean fixation duration
    mean_duration = np.mean(duration)

    # 3. Std fixation duration
    std_duration = np.std(duration)

    # 4. Scanpath length
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    scanpath_length = np.sum(distances)

    # 5. Mean saccade distance
    mean_saccade = np.mean(distances)

    # 6. Spatial dispersion
    dispersion = np.var(x) + np.var(y)

    # 7. Gaze entropy
    heatmap, _, _ = np.histogram2d(x, y, bins=20)
    heatmap = heatmap.flatten()
    heatmap = heatmap[heatmap > 0]
    gaze_entropy = entropy(heatmap)

    return [
        total_fixations,
        mean_duration,
        std_duration,
        scanpath_length,
        mean_saccade,
        dispersion,
        gaze_entropy
    ]


# ---------------------------------
# Build dataset
# ---------------------------------

base_path = "C:/Users/mamil/OneDrive/Documents/autism_detection_project/Data/Saliency4ASD Dataset/Saliency4ASD/dataset"

data = []
labels = []

# ASD files
asd_path = os.path.join(base_path, "ASD")
for file in os.listdir(asd_path):
    if file.endswith(".csv"):
        features = extract_features(os.path.join(asd_path, file))
        if features:
            data.append(features)
            labels.append(1)

# TD files
td_path = os.path.join(base_path, "TD")
for file in os.listdir(td_path):
    if file.endswith(".csv"):
        features = extract_features(os.path.join(td_path, file))
        if features:
            data.append(features)
            labels.append(0)

# Create dataframe
feature_names = [
    "total_fixations",
    "mean_duration",
    "std_duration",
    "scanpath_length",
    "mean_saccade",
    "dispersion",
    "gaze_entropy"
]

dataset = pd.DataFrame(data, columns=feature_names)
dataset["label"] = labels
print(dataset["label"].value_counts())

print(dataset.head())
print("Dataset shape:", dataset.shape)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
# Separate features and labels
X = dataset.drop("label", axis=1)
y = dataset["label"]
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
  )

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(probability=True))
])

param_grid = {
    "classifier__C": [0.1, 1, 10, 50],
    "classifier__kernel": ["rbf"],
    "classifier__gamma": ["scale", "auto"]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nSVM Test Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
# tried too do feature tuning
# from sklearn.model_selection import GridSearchCV
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
#  )
# pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     ("classifier", LogisticRegression(max_iter=5000))
# ])

# param_grid = {
#     "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
#     "classifier__penalty": ["l2"],
#     "classifier__solver": ["lbfgs"]
# }

# grid = GridSearchCV(
#     pipeline,
#     param_grid,
#     cv=5,
#     scoring="f1",
#     n_jobs=-1
# )

# grid.fit(X_train, y_train)

# print("Best Parameters:", grid.best_params_)
# print("Best CV F1:", grid.best_score_)

# best_model = grid.best_estimator_

# y_pred = best_model.predict(X_test)
# y_prob = best_model.predict_proba(X_test)[:, 1]

# print("\nImproved Test Performance:")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("F1:", f1_score(y_test, y_pred))
# print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Build pipeline
# pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     ("classifier", LogisticRegression())
# ])

# # Train model
# pipeline.fit(X_train, y_train)

# # Predictions
# y_pred = pipeline.predict(X_test)
# y_prob = pipeline.predict_proba(X_test)[:, 1]
from sklearn.inspection import permutation_importance

result = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

importance = pd.Series(result.importances_mean, index=X.columns)
print("\nFeature Importance:")
print(importance.sort_values(ascending=False))
# Metrics
print("\n=== Test Set Performance (Eye-Gaze Model) ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")

print("\n=== Cross Validation ===")
print("Mean F1:", cv_scores.mean())
print("Std F1:", cv_scores.std())


# feature_importance = pd.Series(
#     best_model.named_steps["classifier"].coef_[0],
#     index=X.columns
# )

# print(feature_importance.sort_values(ascending=False))
joblib.dump(best_model, "gaze_model.pkl")

