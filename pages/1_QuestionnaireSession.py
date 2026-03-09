
# import streamlit as st
# import pandas as pd
# import joblib

# # --------------------------------------------------
# # Load FINAL Logistic Regression Pipeline
# # --------------------------------------------------
# model = joblib.load("../Models/autism_questionnaire_pipeline.pkl")

# st.set_page_config(page_title="Autism Screening", layout="centered")

# st.title("🧠 Autism Screening Questionnaire")
# st.write(
#     "This short questionnaire helps identify early behavioral patterns that may be associated with autism."
# )

# st.caption("Please answer each question based on your child's typical behavior.There are no right or wrong answers.")

# st.divider()

# # --------------------------------------------------
# # Questionnaire
# # --------------------------------------------------
# questions = {
#     "A1_Score": "Does your child look at you when you call their name?",
#     "A2_Score": "Does your child find it easy to make eye contact?",
#     "A3_Score": "Does your child enjoy playing with other children?",
#     "A4_Score": "Does your child pretend during play?",
#     "A5_Score": "Does your child point to show interest?",
#     "A6_Score": "Does your child respond when spoken to?",
#     "A7_Score": "Does your child show interest in people?",
#     "A8_Score": "Does your child imitate others?",
#     "A9_Score": "Does your child understand simple instructions?",
#     "A10_Score": "Does your child use gestures like waving?"
# }

# responses = {}

# st.subheader("📝 Behavioral Questionnaire")
# for col, text in questions.items():
#     answer = st.selectbox(text, ["Yes", "No"], key=col)
#     # Yes = typical behavior (0 risk), No = deficit (1 risk)
#     responses[col] = 0 if answer == "Yes" else 1

# st.divider()

# # --------------------------------------------------
# # Demographic Inputs
# # --------------------------------------------------
# st.subheader("👶 Background Information")

# age = st.number_input("Age (years)", min_value=1, max_value=18, value=6)

# jundice = st.selectbox("Jaundice at birth?", ["no", "yes"])
# austim = st.selectbox("Family member with ASD?", ["no", "yes"])

# age_desc = st.selectbox("Age Group", ["4-11 years"])

# st.divider()
# # Compute engineered features
# total_deficit_score = sum(responses.values())
# deficit_ratio = total_deficit_score / 10

# if "questionnaire_done" not in st.session_state:
#     st.session_state["questionnaire_done"] = False


# if not st.session_state["questionnaire_done"]:

#     if st.button("Submit Questionnaire"):

#         input_data = {
#             **responses,
#             "age": age,
#             "total_deficit_score": total_deficit_score,
#             "deficit_ratio": deficit_ratio,
#             "jundice": jundice,
#             "austim": austim,
#             "age_desc": age_desc
#         }

#         input_df = pd.DataFrame([input_data])
#         ml_prob = model.predict_proba(input_df)[0][1]

#         st.session_state["q_prob"] = float(ml_prob)
#         st.session_state["questionnaire_done"] = True

#         st.rerun()

# else:
#     st.success("✅ Questionnaire completed successfully.")
#     st.info("Responses recorded. Proceed to eye-gaze assessment.")

#     if st.button("Proceed to Eye-Gaze Assessment"):
#         st.switch_page("pages/2_EyeGazeSession.py")

import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Load Questionnaire Model
# --------------------------------------------------
model = joblib.load("./Models/autism_questionnaire_pipeline.pkl")

st.set_page_config(page_title="Autism Screening Questionnaire", layout="centered")

# --------------------------------------------------
# Page Title
# --------------------------------------------------
st.title("🧠 Autism Screening Questionnaire")

st.write(
    "This short questionnaire helps identify early behavioral patterns "
    "that may be associated with autism."
)

st.caption(
    "Please answer each question based on your child's typical behavior. "
    "There are no right or wrong answers."
)

st.divider()

# --------------------------------------------------
# Instructions for Users
# --------------------------------------------------
st.subheader("📋 Instructions")

st.markdown("""
Please read each question carefully and choose the option that best describes your child's usual behavior.

**Helpful Tips:**

• Answer based on how your child usually behaves in everyday situations.  
• Think about behavior you have observed over the past few months.  
• Select **Yes** if the behavior happens often.  
• Select **No** if the behavior rarely happens or does not occur.  
• If you are unsure, choose the option that best matches what you observe most of the time.

There are **no right or wrong answers**. Your honest responses help provide a more meaningful screening result.
""")

st.info(
    "Take a moment to think about how your child behaves during play, "
    "communication, and interaction with others before answering."
)

st.divider()

# --------------------------------------------------
# Questionnaire
# --------------------------------------------------
st.subheader("📝 Child Behavior Questions")

questions = {
    "A1_Score": "When you call your child's name, do they look at you?",
    "A2_Score": "Does your child make eye contact during interaction?",
    "A3_Score": "Does your child enjoy playing with other children?",
    "A4_Score": "Does your child engage in pretend play (for example, pretending to cook or talk on a toy phone)?",
    "A5_Score": "Does your child point to objects to show interest?",
    "A6_Score": "Does your child respond when someone speaks to them?",
    "A7_Score": "Does your child show interest in interacting with people?",
    "A8_Score": "Does your child try to copy actions or behaviors of others?",
    "A9_Score": "Does your child understand simple instructions?",
    "A10_Score": "Does your child use gestures such as waving goodbye?"
}

responses = {}

for col, text in questions.items():
    answer = st.selectbox(text, ["Yes", "No"], key=col)

    # Yes = typical behavior (0 risk), No = possible deficit (1 risk)
    responses[col] = 0 if answer == "Yes" else 1

st.divider()

# --------------------------------------------------
# Child Information
# --------------------------------------------------
st.subheader("👶 Child Information")

st.markdown(
    "The following information helps provide a more accurate screening result."
)

age = st.number_input("Child's Age (years)", min_value=1, max_value=18, value=6)

jundice = st.selectbox("Did the child have jaundice at birth?", ["no", "yes"])

austim = st.selectbox("Is there a family member diagnosed with autism?", ["no", "yes"])

age_desc = st.selectbox("Age Group", ["4-11 years"])

st.divider()

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
total_deficit_score = sum(responses.values())
deficit_ratio = total_deficit_score / 10

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------
if "questionnaire_done" not in st.session_state:
    st.session_state["questionnaire_done"] = False

# --------------------------------------------------
# Submit Questionnaire
# --------------------------------------------------
if not st.session_state["questionnaire_done"]:

    st.markdown(
        "When you are ready, submit your responses to continue to the next step."
    )

    if st.button("Submit Questionnaire"):

        input_data = {
            **responses,
            "age": age,
            "total_deficit_score": total_deficit_score,
            "deficit_ratio": deficit_ratio,
            "jundice": jundice,
            "austim": austim,
            "age_desc": age_desc
        }

        input_df = pd.DataFrame([input_data])

        # Predict probability
        ml_prob = model.predict_proba(input_df)[0][1]

        # Store result
        st.session_state["q_prob"] = float(ml_prob)
        st.session_state["questionnaire_done"] = True

        st.rerun()

# --------------------------------------------------
# After Submission
# --------------------------------------------------
else:

    st.success("✅ Questionnaire completed successfully.")

    st.info(
        "Thank you for completing the questionnaire.\n\n"
        "In the next step, we will briefly observe how your child looks at images on the screen. "
        "This helps understand patterns of visual attention that may provide additional insight "
        "for the screening."
    )

    if st.button("Continue to Eye Gaze Session"):
        st.switch_page("pages/2_EyeGazeSession.py")
