# import streamlit as st

# st.set_page_config(page_title="Multimodal Fusion Result", layout="centered")

# st.title("🔗 Multimodal ASD Screening Result")
# st.write(
#     "This result combines Questionnaire and Eye-Gaze predictions "
#     "using performance-based decision-level fusion."
# )

# st.divider()

# # ---------------- CHECK REQUIRED DATA ----------------
# if "q_prob" not in st.session_state:
#     st.warning("Questionnaire result not found. Please complete questionnaire first.")
#     st.stop()

# if "probability" not in st.session_state:
#     st.warning("Eye-Gaze result not found. Please complete eye-gaze session first.")
#     st.stop()

# # ---------------- GET STORED PROBABILITIES ----------------
# q_prob = st.session_state["q_prob"]
# g_prob = st.session_state["probability"]

# # ---------------- PERFORMANCE-BASED WEIGHTS ----------------
# alpha = 0.5486   # Questionnaire weight
# beta = 0.4514    # Eye-Gaze weight

# # ---------------- FUSION ----------------
# final_prob = alpha * q_prob + beta * g_prob
# final_percent = final_prob * 100

# # ---------------- RISK LEVEL ----------------
# if final_prob < 0.30:
#     level = "Low Risk"
# elif final_prob < 0.60:
#     level = "Moderate Risk"
# else:
#     level = "High Risk"

# # ---------------- DISPLAY ----------------
# st.subheader("📊 Final Multimodal Screening Result")

# st.metric("Final ASD Probability", f"{final_percent:.2f} %")
# st.progress(float(min(max(final_prob, 0.0), 1.0)))

# st.write(f"**Risk Level:** {level}")

# st.divider()

# # ---------------- BREAKDOWN ----------------
# st.subheader("📌 Contribution Breakdown")

# st.write(f"• Questionnaire Contribution (54.86% weight): {q_prob*100:.2f}%")
# st.write(f"• Eye-Gaze Contribution (45.14% weight): {g_prob*100:.2f}%")

# st.caption(
#     "⚠️ This multimodal system is a screening support tool and not a clinical diagnosis. "
#     "Consult qualified healthcare professionals for medical evaluation."
# )
import streamlit as st

st.set_page_config(page_title="Screening Result", layout="centered")

st.title("🧠 Autism Screening Result")

st.write(
    "Based on the information provided and the visual attention session, "
    "the screening result is shown below."
)

st.divider()

# ---------------- CHECK REQUIRED DATA ----------------
if "q_prob" not in st.session_state:
    st.warning("Questionnaire data not found. Please complete the questionnaire first.")
    st.stop()

if "probability" not in st.session_state:
    st.warning("Visual attention session not found. Please complete the session first.")
    st.stop()

# ---------------- GET STORED PROBABILITIES ----------------
q_prob = st.session_state["q_prob"]
g_prob = st.session_state["probability"]

# ---------------- COMBINED SCREENING SCORE ----------------
alpha = 0.5486
beta = 0.4514

final_prob = alpha * q_prob + beta * g_prob
final_percent = final_prob * 100

# ---------------- RISK LEVEL ----------------
if final_prob < 0.30:
    level = "Low Risk"
elif final_prob < 0.60:
    level = "Moderate Risk"
else:
    level = "High Risk"

# ---------------- RESULT DISPLAY ----------------
st.subheader("📊 Screening Outcome")

st.metric("Screening Score", f"{final_percent:.1f}%")
st.progress(float(min(max(final_prob, 0.0), 1.0)))

st.write(f"### Risk Level: {level}")

st.divider()
# ---------------- BREAKDOWN ----------------
st.subheader("📌 Contribution Breakdown")

st.write(f"• Questionnaire Contribution : {q_prob*100:.2f}%")
st.write(f"• Eye-Gaze Contribution : {g_prob*100:.2f}%")
st.divider()
# ---------------- INTERPRETATION ----------------
st.subheader("What This Means")

if level == "Low Risk":
    st.write(
        "The screening did not detect strong indicators associated with autism. "
        "However, if you have concerns about your child's development, "
        "consider consulting a qualified healthcare professional."
    )

elif level == "Moderate Risk":
    st.write(
        "Some behavioral and visual attention patterns observed during the screening "
        "may benefit from further evaluation by a specialist."
    )

else:
    st.write(
        "The screening detected patterns that may be associated with autism. "
        "It is recommended to consult a qualified healthcare professional "
        "for a comprehensive developmental assessment."
    )

st.divider()

# ---------------- NEXT STEP GUIDANCE ----------------
st.subheader("Next Steps")

st.markdown("""
• This screening tool helps identify possible indicators of autism.  
• It **does not provide a medical diagnosis**.  
• For a full assessment, please consult a qualified healthcare professional such as:

  - Pediatrician  
  - Child psychologist  
  - Developmental specialist
""")


# ---------------- DISCLAIMER ----------------
st.caption(
    "⚠️ This screening tool is intended for informational purposes only "
    "and does not replace professional medical evaluation."
)