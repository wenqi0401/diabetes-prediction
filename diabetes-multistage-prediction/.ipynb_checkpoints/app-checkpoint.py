import streamlit as st
import pandas as pd
import joblib

# ---------- Load saved models (trained already in the notebook) ----------
full_model = joblib.load("models/full_model.pkl")
full_scaler = joblib.load("models/full_scaler.pkl")
reduced_model = joblib.load("models/reduced_model.pkl")
reduced_scaler = joblib.load("models/reduced_scaler.pkl")
le_gender = joblib.load("models/le_gender.pkl")
le_smoking = joblib.load("models/le_smoking.pkl")

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction")
st.write(
    "Compare predictions using only easy-to-know information, or add your lab test "
    "results (HbA1c, blood glucose) for a prediction using the full feature set."
)

def bmi_category(bmi):
    if bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    else:
        return 3

# ---------- User Input ----------
st.header("Your Information")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", le_gender.classes_)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
with col2:
    smoking_history = st.selectbox("Smoking History", le_smoking.classes_)
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)

have_labs = st.checkbox("I have my HbA1c and blood glucose test results")

hba1c, glucose = None, None
if have_labs:
    hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5)
    glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=400, value=100)

if st.button("Predict"):
    gender_enc = le_gender.transform([gender])[0]
    smoking_enc = le_smoking.transform([smoking_history])[0]
    hypertension_val = 1 if hypertension == "Yes" else 0
    heart_disease_val = 1 if heart_disease == "Yes" else 0
    bmi_cat = bmi_category(bmi)

    reduced_input = pd.DataFrame([[
        gender_enc, age, hypertension_val, heart_disease_val, smoking_enc, bmi, bmi_cat
    ]], columns=['gender_enc', 'age', 'hypertension', 'heart_disease', 'smoking_enc', 'bmi', 'bmi_category'])

    reduced_scaled = reduced_scaler.transform(reduced_input)
    reduced_pred = reduced_model.predict(reduced_scaled)[0]
    reduced_proba = reduced_model.predict_proba(reduced_scaled)[0][1]

    st.subheader("Reduced Feature Model (no lab data)")
    if reduced_pred == 1:
        st.error(f"Prediction: **Diabetes Likely** (probability: {reduced_proba:.1%})")
    else:
        st.success(f"Prediction: **No Diabetes** (probability of diabetes: {reduced_proba:.1%})")

    if have_labs:
        full_input = pd.DataFrame([[
            gender_enc, age, hypertension_val, heart_disease_val, smoking_enc, bmi, bmi_cat,
            hba1c, glucose
        ]], columns=['gender_enc', 'age', 'hypertension', 'heart_disease', 'smoking_enc', 'bmi',
                     'bmi_category', 'HbA1c_level', 'blood_glucose_level'])

        full_scaled = full_scaler.transform(full_input)
        full_pred = full_model.predict(full_scaled)[0]
        full_proba = full_model.predict_proba(full_scaled)[0][1]

        st.subheader("Full Feature Model (with lab data)")
        if full_pred == 1:
            st.error(f"Prediction: **Diabetes Likely** (probability: {full_proba:.1%})")
        else:
            st.success(f"Prediction: **No Diabetes** (probability of diabetes: {full_proba:.1%})")

        st.caption(
            "Comparing both results above shows how much the prediction can change "
            "once lab test data is included."
        )
    else:
        st.info("Tick the box above and enter your lab results to also see the full-feature prediction.")

st.divider()
st.caption("This tool is for educational/coursework purposes only and is not a substitute for medical diagnosis.")