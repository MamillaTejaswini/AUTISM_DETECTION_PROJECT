
import joblib
import pandas as pd
import numpy as np
from feature_extraction import extract_features_from_arrays

# ===== Load Model =====
model = joblib.load("gaze_model_enhanced.pkl")

# ===== Load User CSV =====
df = pd.read_csv("user_gaze_data.csv")
df = df.dropna(subset=["gaze_x", "gaze_y"])

x = df["gaze_x"].values
y = df["gaze_y"].values

# Normalize gaze coordinates (0–1)
x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-6)
y = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-6)

timestamps = df["timestamp"].values

# Compute durations
duration = np.diff(timestamps, prepend=timestamps[0])

# Extract features
features_list = extract_features_from_arrays(x, y, duration)

all_feature_names = [
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

# Convert to dataframe
features = pd.DataFrame([features_list], columns=all_feature_names)

print(model.feature_names_in_)

# Match training feature order
features = features[model.feature_names_in_]

# ===== Prediction =====
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0][1]

print("\n=== ASD Prediction Result ===\n")
print(features)

if prediction == 1:
    print("Prediction: ASD")
else:
    print("Prediction: Typical")

print("ASD Probability:", probability)
print("Decision Function:", model.decision_function(features))

print("Scaled features:")
print(model.named_steps["scaler"].transform(features))