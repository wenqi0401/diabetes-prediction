import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_models():
    return {
        "model_a": joblib.load("models/model_a_pipeline.pkl"),
        "model_b": joblib.load("models/model_b_pipeline.pkl"),
    }

models = load_models()
model_a = models["model_a"]
model_b = models["model_b"]

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

st.title("🩺 Diabetes Prediction")
st.write(
    "This application demonstrates two independent prediction models "
    "that use different feature sets."
)

tab_a, tab_b = st.tabs([
    "Model A — Non-Laboratory",
    "Model B — Laboratory-Informed"
])

with tab_a:
    st.header("Model A — Non-Laboratory")
    st.write(
        "Uses only easily obtainable information. "
        "HbA1c and blood glucose results are not required."
    )

    st.info(
        "Input features: Gender, Age, Hypertension, Heart Disease, "
        "Smoking History, and BMI."
    )

    with st.form("model_a_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender_a = st.selectbox(
                "Gender",
                ["Female", "Male", "Other"],
                key="gender_a"
            )
            age_a = st.number_input(
                "Age",
                min_value=0.0,
                max_value=120.0,
                value=30.0,
                step=1.0,
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
                ["No Info", "current", "ever", "former", "never", "not current"],
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
        input_a = pd.DataFrame([{
            "gender": gender_a,
            "age": age_a,
            "hypertension": 1 if hypertension_a == "Yes" else 0,
            "heart_disease": 1 if heart_disease_a == "Yes" else 0,
            "smoking_history": smoking_a,
            "bmi": bmi_a,
        }])

        pred_a = model_a.predict(input_a)[0]
        proba_a = model_a.predict_proba(input_a)[0][1]

        st.subheader("Model A Prediction Result")
        display_prediction(pred_a, proba_a)

with tab_b:
    st.header("Model B — Laboratory-Informed")
    st.write(
        "Uses the same non-laboratory information plus HbA1c and "
        "blood glucose measurements."
    )

    st.info(
        "Input features: Gender, Age, Hypertension, Heart Disease, "
        "Smoking History, BMI, HbA1c Level, and Blood Glucose Level."
    )

    with st.form("model_b_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender_b = st.selectbox(
                "Gender",
                ["Female", "Male", "Other"],
                key="gender_b"
            )
            age_b = st.number_input(
                "Age",
                min_value=0.0,
                max_value=120.0,
                value=30.0,
                step=1.0,
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
                ["No Info", "current", "ever", "former", "never", "not current"],
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
        input_b = pd.DataFrame([{
            "gender": gender_b,
            "age": age_b,
            "hypertension": 1 if hypertension_b == "Yes" else 0,
            "heart_disease": 1 if heart_disease_b == "Yes" else 0,
            "smoking_history": smoking_b,
            "bmi": bmi_b,
            "HbA1c_level": hba1c_b,
            "blood_glucose_level": glucose_b,
        }])

        pred_b = model_b.predict(input_b)[0]
        proba_b = model_b.predict_proba(input_b)[0][1]

        st.subheader("Model B Prediction Result")
        display_prediction(pred_b, proba_b)

st.divider()
st.caption(
    "Model A and Model B are independent comparative models, "
    "not sequential diagnostic stages."
)