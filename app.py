"""Streamlit interface for the student stress classification pipeline."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Student Stress Predictor",
    page_icon="🌿",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "Best_KNN_Pipeline.joblib"
LIKERT_OPTIONS = ["Never", "Almost Never", "Sometimes", "Fairly Often", "Very Often"]
LIKERT_SCORES = {answer: score for score, answer in enumerate(LIKERT_OPTIONS, start=1)}

FEATURES = [
    ("Unexpected_Difficulties", "Unexpected academic difficulties or changes in study plans"),
    ("Academic_Workload", "Amount of academic work feels difficult to manage"),
    ("Deadline_Pressure", "Difficulty completing assignments or projects by their deadlines"),
    ("Examination_Pressure", "Examinations or assessments cause concern about academic performance"),
    ("Career_Pressure", "Placements, internships, employment, or career concerns affect academic life"),
    ("Sleep_Difficulties", "Difficulty getting enough good-quality sleep due to academic routine or concerns"),
    ("Time_Management", "Difficulty allocating enough time to academic responsibilities"),
    ("Lack_of_Control", "Important academic responsibilities feel beyond your control"),
    ("Lack_of_Social_Support", "Lack of support from others when facing academic difficulties"),
    ("Competing_Responsibilities", "Other responsibilities compete with academic responsibilities"),
]

@st.cache_resource
def load_model(path: str, modified_time: float):
    return joblib.load(path)


st.title("Student Stress Predictor")
st.write(
    "Answer the questions based on how often each situation has applied to you. "
    "Your responses are used to estimate a stress category with the project’s selected KNN model."
)

if not MODEL_PATH.is_file():
    st.error("The trained model file is missing, so the predictor cannot run yet.")
    st.markdown(
        "Download **`Best_KNN_Pipeline.joblib`** from your Google Drive "
        "(`ML_Project_Artifacts`) and place it beside `app.py` in this project. "
        "Then commit the model file to the repository used for deployment."
    )
    st.stop()

with st.form("stress_questionnaire"):
    st.subheader("Questionnaire")
    answers = {}
    for feature_name, question in FEATURES:
        answers[feature_name] = st.select_slider(
            question,
            options=LIKERT_OPTIONS,
            value="Never",
            key=feature_name,
        )
    submitted = st.form_submit_button("Estimate stress level", type="primary")

if submitted:
    input_data = pd.DataFrame(
        [[LIKERT_SCORES[answers[name]] for name, _ in FEATURES]],
        columns=[name for name, _ in FEATURES],
    )
    try:
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        prediction = str(model.predict(input_data)[0]).strip().title()
    except Exception as exc:
        st.error(f"Could not load the model or make a prediction: {exc}")
        st.stop()

    total_score = int(input_data.iloc[0].sum())
    stress_percent = (total_score - 10) / (50 - 10) * 100

    result_style = {
        "Low": ("Low stress", "The model classified these responses as Low."),
        "Medium": ("Medium stress", "The model classified these responses as Medium."),
        "High": ("High stress", "The model classified these responses as High."),
    }
    title, detail = result_style.get(
        prediction, (f"{prediction} stress", f"Model prediction: {prediction}")
    )
    st.subheader("KNN model prediction")
    st.success(f"**{title}**\n\n{detail}")
    st.metric("Questionnaire stress percentage", f"{stress_percent:.0f}%")
    st.caption(
        f"The category above comes only from the trained KNN model. The separate percentage "
        f"is calculated from your answers ({total_score}/50 points), normalized from the "
        "minimum score (10) to the maximum (50); it does not determine the category."
    )
    if prediction == "High":
        st.markdown("#### Your highest-scoring stress factors")
        st.write("Based on the answers you entered, these factors received the highest scores:")
        personal_scores = [
            (question, answers[name], LIKERT_SCORES[answers[name]])
            for name, question in FEATURES
        ]
        sorted_scores = sorted(personal_scores, key=lambda item: item[2], reverse=True)
        cutoff = sorted_scores[min(2, len(sorted_scores) - 1)][2]
        displayed_features = [item for item in sorted_scores if item[2] >= cutoff]
        for rank, (feature, _, score) in enumerate(displayed_features, start=1):
            contribution = score / total_score * 100
            st.write(f"{rank}. **{feature}** — {contribution:.1f}% contribution")
        
