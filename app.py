import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Credit Card Approval System",
    page_icon="💳",
    layout="centered"
)

# ===============================
# GLOBAL CSS (BACKGROUND + CARDS)
# ===============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://media.istockphoto.com/id/1203763961/photo/stacked-credit-cards.jpg?s=612x612&w=0&k=20&c=bEEGZwG120WKDClhmltyAtP0kPMzNir49P4JO3pcies=");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Title Card */
.title-card {
    background-color: rgba(0,0,0,0.75);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
}

/* Applicant Card */
.black-card {
    background-color: rgba(0,0,0,0.85);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.6);
}

/* Labels */
.black-card label {
    color: #ffffff !important;
    font-weight: 600;
}

/* Section title */
.section-title {
    text-align: center;
    color: #00FFD1;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 20px;
}

/* Button */
div.stButton > button {
    background-color: #00FFD1;
    color: black;
    font-weight: 700;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL FILES
# ===============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ===============================
# TITLE CARD
# ===============================
st.markdown("""
<div class="title-card">
    <h1 style="color:#00FFD1;">💳 Credit Card Approval Predictor</h1>
    <p style="color:white;font-size:18px;">
        AI-powered decision system using Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ===============================
# INPUT FORM
# ===============================
with st.form("prediction_form"):
    st.markdown('<div class="black-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Applicant Details</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["MALE", "FEMALE"])
        owns_car = st.selectbox("Owns Car", ["YES", "NO"])
        owns_property = st.selectbox("Owns Property", ["YES", "NO"])
        income = st.number_input("Total Income", min_value=0, step=500)

    with col2:
        children = st.number_input("Number of Children", min_value=0, step=1)
        age = st.number_input("Applicant Age", min_value=18, step=1)
        years_working = st.number_input("Years of Working", min_value=0, step=1)
        family_members = st.number_input("Family Members", min_value=1, step=1)

    submit = st.form_submit_button("🔍 Predict Approval")
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# PREDICTION
# ===============================
if submit:
    input_dict = {
        "Applicant_Gender": gender,
        "Owned_Car": owns_car,
        "Owned_Realty": owns_property,
        "Total_Income": income,
        "Total_Children": children,
        "Applicant_Age": age,
        "Years_of_Working": years_working,
        "Total_Family_Members": family_members
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ===============================
    # RESULT UI
    # ===============================
    st.write("")

    if prediction == 1:
        st.markdown(
            f"""
            <div style="background-color:rgba(0,255,180,0.9);
                        padding:20px;
                        border-radius:15px;
                        text-align:center;">
                <h2>✅ APPROVED</h2>
                <h4>Approval Probability: {probability:.2%}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        decision_text = "APPROVED"
    else:
        st.markdown(
            f"""
            <div style="background-color:rgba(255,80,80,0.9);
                        padding:20px;
                        border-radius:15px;
                        text-align:center;">
                <h2>❌ REJECTED</h2>
                <h4>Approval Probability: {probability:.2%}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        decision_text = "REJECTED"

    # ===============================
    # OFFICIAL CREDIT CARD REPORT
    # ===============================
    report_text = f"""
CREDIT CARD APPLICATION DECISION REPORT
---------------------------------------

Decision Status: {decision_text}

After evaluating the applicant’s submitted information through our automated
credit assessment system, the application has been {decision_text.lower()}.

The decision was based on financial stability, employment history, income level,
family responsibility, and asset ownership. The model applies internal credit
risk rules aligned with industry standards.

Approval Probability Score: {probability:.2%}

DIVYANSH DIWAKAR

This report is system-generated for informational purposes only.
"""

    st.write("")
    st.download_button(
        label="📄 Download Official Decision Report",
        data=report_text,
        file_name="Credit_Card_Decision_Report.txt",
        mime="text/plain"
    )