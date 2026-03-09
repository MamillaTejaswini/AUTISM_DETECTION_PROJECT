import os
import pandas as pd
import numpy as np
from scipy.stats import entropy, skew, kurtosis
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
from feature_extraction import extract_features_from_arrays
from sklearn.feature_selection import SelectKBest, f_classif
base_path = "C:/Users/mamil/OneDrive/Documents/autism_detection_project/Data/Saliency4ASD Dataset/Saliency4ASD/dataset"

data = []
labels = []

for label_name, label_value in [("ASD", 1), ("TD", 0)]:
    folder = os.path.join(base_path, label_name)
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            df.columns = df.columns.str.strip()
            x = df["x"].values
            y = df["y"].values
            x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-6)
            y = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-6)
            duration = df["duration"].values
            features = extract_features_from_arrays(x, y, duration)
            if features:
                data.append(features)
                labels.append(label_value)

feature_names = [

"scanpath_length",
"mean_saccade",
"std_saccade",
"dispersion",
"gaze_entropy",

"velocity_mean",
"velocity_std",
"velocity_max",

"var_ratio",
"fixation_density",

"mean_duration",
"std_duration",
"duration_skew",
"duration_kurt",
"long_fix_ratio",

"center_distance_mean",
"gaze_instability",
"unique_cells",
"path_efficiency"

]

dataset_full = pd.DataFrame(data, columns=[

"scanpath_length",
"mean_saccade",
"std_saccade",
"dispersion",
"gaze_entropy",

"velocity_mean",
"velocity_std",
"velocity_max",

"var_ratio",
"fixation_density",

"mean_duration",
"std_duration",
"duration_skew",
"duration_kurt",
"long_fix_ratio",

"center_distance_mean",
"gaze_instability",
"unique_cells",
"path_efficiency"

])


dataset = dataset_full.copy()
dataset = dataset.copy()
dataset["label"] = labels

print("Dataset shape:", dataset.shape)

X = dataset.drop("label", axis=1)
y = dataset["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------
# SVM with GridSearch
# ---------------------------------

# pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     ("classifier", SVC(probability=True))
# ])
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("feature_selection", SelectKBest(score_func=f_classif, k=15)),
    ("classifier", SVC(probability=True,class_weight="balanced"))
])
param_grid = {
    "classifier__C": [0.1, 1, 10, 50],
    "classifier__gamma": ["scale", "auto"],
    "classifier__kernel": ["linear"]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]
print("Number of support vectors:", best_model.named_steps['classifier'].n_support_)
print("Dual Coefficients shape:", best_model.named_steps['classifier'].dual_coef_.shape)
print("\nEnhanced SVM Test Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="f1")

print("\nCross Validation F1:", cv_scores.mean(), "±", cv_scores.std())
selector = best_model.named_steps["feature_selection"]
selected_features = X.columns[selector.get_support()]

print("\nSelected Features:")
print(selected_features)
joblib.dump(best_model, "gaze_model_enhanced.pkl")
print(best_model.feature_names_in_)    