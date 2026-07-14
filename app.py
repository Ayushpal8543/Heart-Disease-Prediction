import streamlit as st
import pandas as pd
import joblib
import os

from src.recommendation import get_recommendation
from src.utils import generate_pdf

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")


st.title("❤️ Heart Disease Prediction System")
st.markdown("Predict the **10-Year Risk of Coronary Heart Disease (CHD)**")

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("About")

st.sidebar.info("""
Machine Learning Models Used

- Logistic Regression
- Random Forest
- GridSearchCV

Dataset

- Framingham Heart Study

Developed using

- Python
- Scikit-Learn
- Streamlit
""")

# -----------------------------
# Input Form
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    
    patient_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient's name"
    )

    male = st.selectbox("Gender", ["Female", "Male"])

    age = st.slider("Age", 20, 80, 45)

    education_label = st.selectbox(
        "Education Level",
        [
            "Some High School",
            "High School Graduate",
            "Some College",
            "College Graduate"
        ]
    )

    education_map = {
        "Some High School": 1,
        "High School Graduate": 2,
        "Some College": 3,
        "College Graduate": 4
    }

    education = education_map[education_label]

    smoker = st.selectbox(
        "Current Smoker",
        ["No","Yes"]
    )

    cigs = st.slider(
        "Cigarettes Per Day",
        0,
        70,
        0
    )

    bpmeds = st.selectbox(
        "BP Medication",
        ["No","Yes"]
    )

    stroke = st.selectbox(
        "Previous Stroke",
        ["No","Yes"]
    )

with col2:

    hypertension = st.selectbox(
        "Hypertension",
        ["No","Yes"]
    )

    diabetes = st.selectbox(
        "Diabetes",
        ["No","Yes"]
    )

    chol = st.number_input(
        "Total Cholesterol",
        100,
        700,
        200
    )

    sysBP = st.number_input(
        "Systolic BP",
        80,
        300,
        120
    )

    diaBP = st.number_input(
        "Diastolic BP",
        40,
        200,
        80
    )

    bmi = st.number_input(
        "BMI",
        10.0,
        60.0,
        25.0
    )

    heartRate = st.number_input(
        "Heart Rate",
        40,
        180,
        72
    )

    glucose = st.number_input(
        "Glucose",
        40,
        400,
        90
    )

# -----------------------------
# Predict
# -----------------------------

if st.button("❤️ Predict Heart Disease Risk"):

    # Education Mapping
    

    patient = pd.DataFrame([{
        
        "Patient Name": patient_name,
        "male": 1 if male == "Male" else 0,
        "age": age,
        "education": education,
        "currentSmoker": 1 if smoker == "Yes" else 0,
        "cigsPerDay": cigs,
        "BPMeds": 1 if bpmeds == "Yes" else 0,
        "prevalentStroke": 1 if stroke == "Yes" else 0,
        "prevalentHyp": 1 if hypertension == "Yes" else 0,
        "diabetes": 1 if diabetes == "Yes" else 0,
        "totChol": chol,
        "sysBP": sysBP,
        "diaBP": diaBP,
        "BMI": bmi,
        "heartRate": heartRate,
        "glucose": glucose

    }])
    patient_input = patient.drop(columns=["Patient Name"])
    prediction = model.predict(patient_input)[0]
    probability = model.predict_proba(patient_input)[0][1]
    
    report = get_recommendation(probability)
    generate_pdf(
       patient,
       probability,
       report["level"],
       report["recommendations"]
    )

    st.divider()

    st.subheader("🩺 Risk Assessment")

    risk_level = report["level"]
    
    if "Very Low" in risk_level or "Low" in risk_level:
        st.success(f"{risk_level} ({probability*100:.2f}%)")
    
    elif "Moderate" in risk_level:
        st.warning(f"{risk_level} ({probability*100:.2f}%)")
    
    else:
        st.error(f"{risk_level} ({probability*100:.2f}%)")

    st.subheader("Risk Probability")

    st.progress(float(probability))

    st.metric(
        label="10-Year Heart Disease Risk",
        value=f"{probability*100:.2f}%"
    )

    st.subheader("💡 Health Recommendations")

    for rec in report["recommendations"]:
        st.write("✔", rec)
    

# -----------------------------
# Download PDF Report
# -----------------------------
    if os.path.exists("outputs/patient_report.pdf"):
    
        with open("outputs/patient_report.pdf", "rb") as pdf:
    
            st.download_button(
                label="📄 Download Patient Report",
                data=pdf,
                file_name="patient_report.pdf",
                mime="application/pdf"
            )
# -----------------------------
# Model Visualizations
# -----------------------------

st.divider()

st.header("📊 Model Visualizations")

col1, col2 = st.columns(2)

with col1:

    if os.path.exists("images/roc_curve.png"):

        st.image(
            "images/roc_curve.png",
            caption="ROC Curve",
            use_container_width=True
        )

    else:

        st.warning("ROC Curve image not found.")

with col2:

    if os.path.exists("images/feature_importance.png"):

        st.image(
            "images/feature_importance.png",
            caption="Feature Importance",
            use_container_width=True
        )

    else:

        st.warning("Feature Importance image not found.")