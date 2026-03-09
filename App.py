import streamlit as st

st.set_page_config(
    page_title="Autism Screening System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- STYLE --------------------
st.markdown("""
<style>
.title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
}
.subtitle {
    font-size: 20px;
    text-align: center;
    color: #6c757d;
}
.section-title {
    font-size: 24px;
    font-weight: 600;
}
.box {
    background-color: #f4f6f9;
    padding: 20px;
    border-radius: 12px;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("<div class='title'>🧠 Autism Risk Screening</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Early Behavioral & Visual Attention Screening Tool</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------- ABOUT --------------------
st.markdown("<div class='section-title'>About This Tool</div>", unsafe_allow_html=True)

st.markdown("""
<div class='box'>
This system provides early autism risk screening by analyzing:

• Behavioral responses to structured questions  
• Visual attention patterns using eye tracking  

The results help identify potential risk indicators that may require
further professional evaluation.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------- HOW IT WORKS --------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='section-title'>How It Works</div>", unsafe_allow_html=True)
    st.markdown("""
    1. Complete a short behavioral questionnaire  
    2. Participate in a brief eye-tracking session  
    3. Receive an overall screening result  
    """)

with col2:
    st.markdown("<div class='section-title'>Before You Begin</div>", unsafe_allow_html=True)
    st.markdown("""
    • Ensure webcam access is enabled  
    • Sit comfortably facing the screen  
    • Maintain good lighting conditions  
    • Answer all questions honestly  
    """)

st.markdown("<br><br>", unsafe_allow_html=True)

# -------------------- START BUTTON --------------------
center = st.columns([1,2,1])
with center[1]:
    # if st.button("Start Screening", use_container_width=True):
    #     st.switch_page("pages/1_QuestionnaireSession.py")
    if st.button("Start Screening", use_container_width=True):

    # Reset all stored results
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.switch_page("pages/1_QuestionnaireSession.py")

st.markdown("<br><br>", unsafe_allow_html=True)

# -------------------- DISCLAIMER --------------------
st.markdown(
    "<div class='footer'>⚠ This tool is intended for screening support only. It does not provide a medical diagnosis. Please consult a qualified healthcare professional for clinical assessment.</div>",
    unsafe_allow_html=True
)