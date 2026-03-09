import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix,precision_score,recall_score
from xgboost import XGBClassifier
import joblib

def extract_features(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if len(df) < 2:
        return None

    x = df["x"].values
    y = df["y"].values
    duration = df["duration"].values

    total_fixations = len(df)
    mean_duration = np.mean(duration)
    std_duration = np.std(duration)

    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    scanpath_length = np.sum(distances)
    mean_saccade = np.mean(distances)

    dispersion = np.var(x) + np.var(y)

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

base_path = "C:/Users/mamil/OneDrive/Documents/autism_detection_project/Data/Saliency4ASD Dataset/Saliency4ASD/dataset"

data = []
labels = []

for label_name, label_value in [("ASD", 1), ("TD", 0)]:
    folder = os.path.join(base_path, label_name)
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            features = extract_features(os.path.join(folder, file))
            if features:
                data.append(features)
                labels.append(label_value)

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

X = dataset.drop("label", axis=1)
y = dataset["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2]
}

grid = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nXGBoost Test Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="f1")

print("\nCross Validation F1:", cv_scores.mean(), "±", cv_scores.std())

joblib.dump(best_model, "gaze_xgboost.pkl")