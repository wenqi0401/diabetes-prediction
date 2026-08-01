import streamlit as st
import pandas as pd
import joblib

# ---------- Load saved models (trained already in the notebook) ----------
stage1_model = joblib.load("models/stage1_model.pkl")
stage1_scaler = joblib.load("models/stage1_scaler.pkl")
stage2_model = joblib.load("models/stage2_model.pkl")
stage2_scaler = joblib.load("models/stage2_scaler.pkl")
le_gender = joblib.load("models/le_gender.pkl")
le_smoking = joblib.load("models/le_smoking.pkl")

st.set_page_config(page_title="Diabetes Multistage Prediction", layout="centered")
st.title("🩺 Diabetes Multistage Prediction")
st.write("Stage 1 screens your risk without lab tests. Stage 2 gives a clinical classification using lab values.")

# ---------- User Input ----------
st.header("Patient Information")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", le_gender.classes_)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
with col2:
    smoking_history = st.selectbox("Smoking History", le_smoking.classes_)
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)

def bmi_category(bmi):
    if bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    else:
        return 3

# ---------- Stage 1 Prediction ----------
if st.button("Run Stage 1: Risk Screening"):
    gender_enc = le_gender.transform([gender])[0]
    smoking_enc = le_smoking.transform([smoking_history])[0]
    hypertension_val = 1 if hypertension == "Yes" else 0
    heart_disease_val = 1 if heart_disease == "Yes" else 0
    bmi_cat = bmi_category(bmi)

    stage1_input = pd.DataFrame([[
        gender_enc, age, hypertension_val, heart_disease_val,
        smoking_enc, bmi, bmi_cat
    ]], columns=['gender_enc', 'age', 'hypertension', 'heart_disease',
                 'smoking_enc', 'bmi', 'bmi_category'])

    stage1_scaled = stage1_scaler.transform(stage1_input)
    stage1_pred = stage1_model.predict(stage1_scaled)[0]

    st.session_state["stage1_done"] = True
    st.session_state["stage1_result"] = "High Risk" if stage1_pred == 1 else "Low Risk"
    st.session_state["stage1_inputs"] = (gender_enc, age, hypertension_val, heart_disease_val, smoking_enc, bmi, bmi_cat)

if st.session_state.get("stage1_done"):
    result = st.session_state["stage1_result"]
    if result == "High Risk":
        st.error(f"Stage 1 Result: **{result}**")
    else:
        st.success(f"Stage 1 Result: **{result}**")

    # ---------- Stage 2 (only if user wants to continue) ----------
    st.header("Stage 2: Clinical Classification")
    st.write("Enter lab values to get a more detailed classification.")

    hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5)
    glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=400, value=100)

    if st.button("Run Stage 2: Clinical Classification"):
        gender_enc, age, hypertension_val, heart_disease_val, smoking_enc, bmi, bmi_cat = st.session_state["stage1_inputs"]

        stage2_input = pd.DataFrame([[
            gender_enc, age, hypertension_val, heart_disease_val,
            smoking_enc, bmi, bmi_cat, hba1c, glucose
        ]], columns=['gender_enc', 'age', 'hypertension', 'heart_disease',
                     'smoking_enc', 'bmi', 'bmi_category', 'HbA1c_level', 'blood_glucose_level'])

        stage2_scaled = stage2_scaler.transform(stage2_input)
        stage2_pred = stage2_model.predict(stage2_scaled)[0]

        labels = {0: "Non-diabetic", 1: "Prediabetic", 2: "Diabetic"}
        result2 = labels[stage2_pred]

        if result2 == "Diabetic":
            st.error(f"Stage 2 Result: **{result2}**")
        elif result2 == "Prediabetic":
            st.warning(f"Stage 2 Result: **{result2}**")
        else:
            st.success(f"Stage 2 Result: **{result2}**")

st.divider()
st.caption("This tool is for educational/coursework purposes only and is not a substitute for medical diagnosis.")
