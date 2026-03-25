import sys
import os
import pandas as pd
import streamlit as st

# ---------------- PATH FIX ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "src"))

from feature_engineering import predict

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Job Offer Predictor",
    page_icon="🚀",
    layout="wide"
)

# ---------------- FINAL UI ----------------
st.markdown("""
<style>

/* ===== GLOBAL ===== */
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e5e7eb;
}

/* ===== CONTAINER FIX ===== */
.block-container {
    padding-top: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1400px;
    margin: auto;
}

/* ===== TITLE ===== */
.title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    color: #14b8a6;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 35px;
}

/* ===== CARDS ===== */
.card {
    background: rgba(15, 23, 42, 0.85);
    padding: 24px;
    border-radius: 18px;
    border: 1px solid rgba(20,184,166,0.15);
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    margin-bottom: 20px;
}

/* ===== SECTION ===== */
.section-title {
    font-size: 20px;
    font-weight: 600;
    color: #f9fafb;
    margin-bottom: 14px;
}

/* ===== SELECT ===== */
.stSelectbox div[data-baseweb="select"] {
    background-color: #020617;
    border-radius: 10px;
}

/* ===== SLIDER ===== */
.stSlider > div > div > div > div {
    background-color: #14b8a6 !important;
}

.stSlider div[role="slider"] {
    background-color: #14b8a6 !important;
    border: 2px solid #0f172a !important;
}

/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(90deg, #14b8a6, #0ea5e9);
    color: white;
    border-radius: 12px;
    height: 3.2em;
    font-size: 16px;
    font-weight: 600;
    width: 100%;
    border: none;
}

/* ===== RESULT CARD ===== */
.result-card {
    margin: 60px auto;
    width: 520px;
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid rgba(20,184,166,0.2);
    box-shadow: 0 20px 60px rgba(0,0,0,0.8);
}

.result-percent {
    font-size: 72px;
    font-weight: 900;
    color: #14b8a6;
    margin-bottom: 10px;
}

.result-label {
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 20px;
}

.result-insight {
    font-size: 15px;
    color: #cbd5f5;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🧑🏻‍💻💼 JOB OFFER PREDICTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered prediction based on student profile & job search behavior</div>', unsafe_allow_html=True)

# ---------------- LAYOUT ----------------
left, right = st.columns([1.1, 1])

# -------- LEFT --------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎓 Academic Profile</div>', unsafe_allow_html=True)

    University_Rating = st.selectbox("University", ["Top-tier", "Mid-tier", "Lower-tier"])
    Major_Category = st.selectbox("Major", ["STEM", "Business", "Arts", "Healthcare", "Humanities"])
    School_Size = st.selectbox("School Size", ["Small", "Medium", "Large"])
    Region = st.selectbox("Region", ["West", "Midwest", "South", "Northeast"])

    GPA = st.slider("GPA", 2.0, 4.0, 3.0)
    Prior_Internships = st.slider("Internships", 0, 5, 1)

    st.markdown('</div>', unsafe_allow_html=True)

# -------- RIGHT --------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Job Search Behavior</div>', unsafe_allow_html=True)

    Primary_Search_Platform = st.selectbox("Platform", ["LinkedIn", "Handshake", "Indeed"])
    Extra_Curricular_Activities = st.slider("Activities", 0, 10, 2)
    Networking_Events_Attended = st.slider("Networking Events", 0, 15, 3)
    Months_Searching = st.slider("Months Searching", 1, 12, 6)
    Applications_Submitted = st.slider("Applications", 5, 300, 50)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Outcome"):

    input_data = pd.DataFrame([{
        "University_Rating": University_Rating,
        "School_Size": School_Size,
        "Region": Region,
        "Major_Category": Major_Category,
        "GPA": GPA,
        "Prior_Internships": Prior_Internships,
        "Extra_Curricular_Activities": Extra_Curricular_Activities,
        "Networking_Events_Attended": Networking_Events_Attended,
        "Primary_Search_Platform": Primary_Search_Platform,
        "Months_Searching": Months_Searching,
        "Applications_Submitted": Applications_Submitted
    }])

    pred, prob = predict(input_data)
    confidence = prob[0] * 100

    # -------- INSIGHTS --------
    insights = []

    if GPA >= 3.5:
        insights.append("High GPA is boosting your chances")
    elif GPA < 2.5:
        insights.append("Low GPA is hurting your chances")

    if Applications_Submitted > 100:
        insights.append("High number of applications improves visibility")
    elif Applications_Submitted < 20:
        insights.append("Too few applications submitted")

    if Networking_Events_Attended > 5:
        insights.append("Strong networking activity helps a lot")
    elif Networking_Events_Attended == 0:
        insights.append("No networking is a major drawback")

    if Prior_Internships >= 2:
        insights.append("Internship experience is a strong advantage")

    insight_text = "<br>".join(insights[:3]) if insights else "No strong signals detected"
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-percent">{confidence:.0f}%</div>
            <div class="result-label">Chance of Job Offer</div>
            <div class="result-insight">{insight_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")