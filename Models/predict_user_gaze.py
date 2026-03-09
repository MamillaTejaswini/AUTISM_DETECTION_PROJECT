import joblib
import numpy as np
import pandas as pd
# Load trained model
model = joblib.load("gaze_model.pkl")

# Replace these numbers with your live output

features = pd.DataFrame([[
    723,
    1.0,
    0.0,
    5608.8478,
    7.7684,
    696.1785,
    4.31038
]], columns=[
    "total_fixations",
    "mean_duration",
    "std_duration",
    "scanpath_length",
    "mean_saccade",
    "dispersion",
    "gaze_entropy"
])

prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0][1]

print("\n=== ASD Gaze Prediction ===\n")

if prediction == 1:
    print("Prediction: ASD-like gaze pattern")
else:
    print("Prediction: Typical gaze pattern")

print("ASD Probability:", probability)