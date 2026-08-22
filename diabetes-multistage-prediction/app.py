import streamlit as st
import pandas as pd
import joblib

# ---------- Page configuration ----------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# ---------- Load saved models ----------
@st.cache_resource
def load_models():
    return {
        "full_model": joblib.load("models/full_model.pkl"),
        "full_scaler": joblib.load("models/full_scaler.pkl"),
        "reduced_model": joblib.load("models/reduced_model.pkl"),
        "reduced_scaler": joblib.load("models/reduced_scaler.pkl"),
        "le_gender": joblib.load("models/le_gender.pkl"),
        "le_smoking": joblib.load("models/le_smoking.pkl"),
    }

models = load_models()

full_model = models["full_model"]
full_scaler = models["full_scaler"]
reduced_model = models["reduced_model"]
reduced_scaler = models["reduced_scaler"]
le_gender = models["le_gender"]
le_smoking = models["le_smoking"]


# ---------- Helper functions ----------
def bmi_category(bmi):
    if bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    else:
        return 3


def encode_common_inputs(gender, age, hypertension, heart_disease, smoking_history, bmi):
    gender_enc = le_gender.transform([gender])[0]
    smoking_enc = le_smoking.transform([smoking_history])[0]
    hypertension_val = 1 if hypertension == "Yes" else 0
    heart_disease_val = 1 if heart_disease == "Yes" else 0
    bmi_cat = bmi_category(bmi)

    return (
        gender_enc,
        age,
        hypertension_val,
        heart_disease_val,
        smoking_enc,
        bmi,
        bmi_cat,
    )


def display_prediction(prediction, probability):
    if prediction == 1:
        st.error(
            f"Prediction: **Diabetes Likely**  \n"
            f"Probability of diabetes: **{probability:.1%}**"
        )
    else:
        st.success(
            f"Prediction: **No Diabetes**  \n"
            f"Probability of diabetes: **{probability:.1%}**"
        )


# ---------- Header ----------
st.title("🩺 Diabetes Prediction")
st.write(
    "This application provides two independent prediction models. "
    "Choose the tab that matches the information you have available."
)

# ---------- Separate model tabs ----------
tab_a, tab_b = st.tabs([
    "Model A — Non-Laboratory",
    "Model B — Laboratory-Informed"
])


# =========================================================
# MODEL A
# =========================================================
with tab_a:
    st.header("Model A — Non-Laboratory Screening")
    st.write(
        "This model uses only easily obtainable, non-laboratory information. "
        "HbA1c and blood glucose test results are **not required**."
    )

    st.info(
        "Input features: Gender, Age, Hypertension, Heart Disease, "
        "Smoking History, BMI, and BMI Category."
    )

    with st.form("model_a_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender_a = st.selectbox(
                "Gender",
                le_gender.classes_,
                key="gender_a"
            )
            age_a = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=30,
                key="age_a"
            )
            hypertension_a = st.selectbox(
                "Hypertension",
                ["No", "Yes"],
                key="hypertension_a"
            )

        with col2:
            heart_disease_a = st.selectbox(
                "Heart Disease",
                ["No", "Yes"],
                key="heart_disease_a"
            )
            smoking_a = st.selectbox(
                "Smoking History",
                le_smoking.classes_,
                key="smoking_a"
            )
            bmi_a = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=70.0,
                value=25.0,
                step=0.1,
                key="bmi_a"
            )

        predict_a = st.form_submit_button(
            "Predict with Model A",
            use_container_width=True
        )

    if predict_a:
        (
            gender_enc,
            age_val,
            hypertension_val,
            heart_disease_val,
            smoking_enc,
            bmi_val,
            bmi_cat,
        ) = encode_common_inputs(
            gender_a,
            age_a,
            hypertension_a,
            heart_disease_a,
            smoking_a,
            bmi_a,
        )

        reduced_input = pd.DataFrame(
            [[
                gender_enc,
                age_val,
                hypertension_val,
                heart_disease_val,
                smoking_enc,
                bmi_val,
                bmi_cat,
            ]],
            columns=[
                "gender_enc",
                "age",
                "hypertension",
                "heart_disease",
                "smoking_enc",
                "bmi",
                "bmi_category",
            ],
        )

        reduced_scaled = reduced_scaler.transform(reduced_input)
        reduced_pred = reduced_model.predict(reduced_scaled)[0]
        reduced_proba = reduced_model.predict_proba(reduced_scaled)[0][1]

        st.subheader("Model A Prediction Result")
        display_prediction(reduced_pred, reduced_proba)

        st.caption(
            "Model A predicts diabetes risk using the reduced feature set only."
        )


# =========================================================
# MODEL B
# =========================================================
with tab_b:
    st.header("Model B — Laboratory-Informed Screening")
    st.write(
        "This model uses the full feature set, including HbA1c and blood glucose "
        "test results in addition to the non-laboratory information."
    )

    st.info(
        "Input features: Gender, Age, Hypertension, Heart Disease, Smoking History, "
        "BMI, BMI Category, HbA1c Level, and Blood Glucose Level."
    )

    with st.form("model_b_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender_b = st.selectbox(
                "Gender",
                le_gender.classes_,
                key="gender_b"
            )
            age_b = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=30,
                key="age_b"
            )
            hypertension_b = st.selectbox(
                "Hypertension",
                ["No", "Yes"],
                key="hypertension_b"
            )
            hba1c_b = st.number_input(
                "HbA1c Level (%)",
                min_value=3.0,
                max_value=15.0,
                value=5.5,
                step=0.1,
                key="hba1c_b"
            )

        with col2:
            heart_disease_b = st.selectbox(
                "Heart Disease",
                ["No", "Yes"],
                key="heart_disease_b"
            )
            smoking_b = st.selectbox(
                "Smoking History",
                le_smoking.classes_,
                key="smoking_b"
            )
            bmi_b = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=70.0,
                value=25.0,
                step=0.1,
                key="bmi_b"
            )
            glucose_b = st.number_input(
                "Blood Glucose Level (mg/dL)",
                min_value=50,
                max_value=400,
                value=100,
                key="glucose_b"
            )

        predict_b = st.form_submit_button(
            "Predict with Model B",
            use_container_width=True
        )

    if predict_b:
        (
            gender_enc,
            age_val,
            hypertension_val,
            heart_disease_val,
            smoking_enc,
            bmi_val,
            bmi_cat,
        ) = encode_common_inputs(
            gender_b,
            age_b,
            hypertension_b,
            heart_disease_b,
            smoking_b,
            bmi_b,
        )

        full_input = pd.DataFrame(
            [[
                gender_enc,
                age_val,
                hypertension_val,
                heart_disease_val,
                smoking_enc,
                bmi_val,
                bmi_cat,
                hba1c_b,
                glucose_b,
            ]],
            columns=[
                "gender_enc",
                "age",
                "hypertension",
                "heart_disease",
                "smoking_enc",
                "bmi",
                "bmi_category",
                "HbA1c_level",
                "blood_glucose_level",
            ],
        )

        full_scaled = full_scaler.transform(full_input)
        full_pred = full_model.predict(full_scaled)[0]
        full_proba = full_model.predict_proba(full_scaled)[0][1]

        st.subheader("Model B Prediction Result")
        display_prediction(full_pred, full_proba)

        st.caption(
            "Model B predicts diabetes risk using the full laboratory-informed feature set."
        )


# ---------- Footer ----------
st.divider()