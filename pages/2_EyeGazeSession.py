# import streamlit as st
# import subprocess
# import pandas as pd
# import joblib
# import os
# import sys

# # Add Models folder
# sys.path.append(os.path.abspath("../Models"))
# from feature_extraction import extract_features_from_arrays

# MODEL_PATH = "C:/Users/mamil/OneDrive/Documents/autism_detection_project/Models/gaze_model_enhanced.pkl"
# GAZE_CSV = "C:/Users/mamil/OneDrive/Documents/autism_detection_project/Models/user_gaze_data.csv"

# st.title("Eye Gaze Session")

# if st.button("Start Eye Tracking Session"):

#     st.info("Launching eye tracking window...")
#     subprocess.Popen(
#     ["python", "../Models/eyetracking_step2.py"],
#     creationflags=subprocess.CREATE_NEW_CONSOLE
# ).wait()

#     if not os.path.exists(GAZE_CSV):
#         st.error("No gaze data found.")
#         st.stop()

#     df = pd.read_csv(GAZE_CSV)

#     if len(df) < 20:
#         st.error("Not enough gaze data collected.")
#         st.stop()

#     model = joblib.load(MODEL_PATH)

#     x = df["gaze_x"].values
#     y = df["gaze_y"].values
#     timestamps = df["timestamp"].values

#     import numpy as np
#     duration = np.diff(timestamps, prepend=timestamps[0])

#     features_list = extract_features_from_arrays(x, y, duration)

#     features = pd.DataFrame(
#         [features_list],
#         columns=model.feature_names_in_
#     )

#     prediction = model.predict(features)[0]
#     probability = model.predict_proba(features)[0][1]

#     # Add this line
#     decision_score = model.decision_function(features)[0]

#     st.session_state["prediction"] = prediction
#     st.session_state["probability"] = float(probability)
#     st.session_state["decision_score"] = float(decision_score)
#     st.session_state["gaze_dataframe"] = df

#     st.switch_page("pages/3_FinalResult.py")



import streamlit as st
import subprocess
import pandas as pd
import joblib
import os
import sys
import numpy as np

# Add Models folder
sys.path.append(os.path.abspath("./Models"))
from feature_extraction import extract_features_from_arrays

MODEL_PATH = "C:/Users/mamil/OneDrive/Documents/autism_detection_project/Models/gaze_model_enhanced.pkl"
GAZE_CSV = "C:/Users/mamil/OneDrive/Documents/autism_detection_project/Models/user_gaze_data.csv"

st.set_page_config(page_title="Visual Attention Session", layout="centered")

# --------------------------------------------------
# Title and Description
# --------------------------------------------------
st.title("👀 Eye Gaze Session")

st.write(
    "In this step, the system observes how your child naturally looks at images "
    "on the screen. This helps identify patterns in visual attention that may "
    "provide additional insight for autism screening."
)

st.divider()

# --------------------------------------------------
# Calibration Explanation
# --------------------------------------------------
st.subheader("🎯 Camera Calibration")

st.markdown("""
Before the session begins, the system briefly adjusts to your child's eye position.

This process helps the system understand where your child is looking on the screen.
For accurate results, it is important that the child's face is clearly visible to the camera.
""")

st.divider()

# --------------------------------------------------
# Setup Instructions
# --------------------------------------------------
st.subheader("📋 Before Starting")

st.markdown("""
Please make sure the following conditions are met:

**Camera Setup**
• The webcam is enabled  
• Your child's face is clearly visible to the camera  
• The child is sitting about **40–70 cm** from the screen  

**Environment**
• Ensure the room has **good lighting**  
• Avoid strong light directly behind the child  

**During the Session**
• Ask your child to look naturally at the images shown  
• The child does **not need to stare at any specific point**  
• Try to minimize large head movements  

The session usually takes **30–60 seconds**.
""")

st.info(
    "When you are ready, click the button below to begin the visual attention session."
)

st.divider()

# --------------------------------------------------
# Start Eye Tracking
# --------------------------------------------------
if st.button("Start Eye Gaze Session"):

    st.info("Launching eye tracking window... Please keep the child facing the screen.")

    subprocess.Popen(
        ["python", "./Models/eyetracking_step2.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    ).wait()

    # --------------------------------------------------
    # Check gaze data file
    # --------------------------------------------------
    if not os.path.exists(GAZE_CSV):
        st.error("No gaze data was recorded. Please try the session again.")
        st.stop()

    df = pd.read_csv(GAZE_CSV)

    if len(df) < 20:
        st.error(
            "Not enough gaze data was collected. "
            "Please ensure the child remains visible to the camera and try again."
        )
        st.stop()

    # --------------------------------------------------
    # Load Model
    # --------------------------------------------------
    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------
    # Extract gaze coordinates
    # --------------------------------------------------
    x = df["gaze_x"].values
    y = df["gaze_y"].values
    timestamps = df["timestamp"].values

    duration = np.diff(timestamps, prepend=timestamps[0])

    # --------------------------------------------------
    # Feature Extraction
    # --------------------------------------------------
    features_list = extract_features_from_arrays(x, y, duration)

    features = pd.DataFrame(
        [features_list],
        columns=model.feature_names_in_
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    decision_score = model.decision_function(features)[0]

    # --------------------------------------------------
    # Store Results
    # --------------------------------------------------
    st.session_state["prediction"] = prediction
    st.session_state["probability"] = float(probability)
    st.session_state["decision_score"] = float(decision_score)
    st.session_state["gaze_dataframe"] = df

    # --------------------------------------------------
    # Move to final result page
    # --------------------------------------------------
    st.switch_page("pages/3_FinalResult.py")