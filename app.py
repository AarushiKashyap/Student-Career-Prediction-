
import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessor
xgb_model = joblib.load("student_placement_xgboost.pkl")
preprocessor = joblib.load("student_placement_preprocessor.pkl")

# Page configuration
st.set_page_config(
    page_title="Student Career Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Career Prediction System")
st.write(
    "Enter your academic, technical and personal details "
    "to predict your placement status."
)

st.info("Please enter all 23 student parameters.")

with st.form("student_prediction_form"):

    st.header("1. Academic Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 40, 22)
        cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0, step=0.1)
        branch = st.selectbox(
            "Branch",
            ["CSE", "IT", "ECE", "EEE", "Civil", "Mechanical"]
        )

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        college_tier = st.selectbox(
            "College Tier", ["Tier 1", "Tier 2", "Tier 3"]
        )
        attendance_percentage = st.number_input(
            "Attendance Percentage", 0.0, 100.0, 80.0
        )

    with col3:
        backlogs = st.number_input("Backlogs", 0, 20, 0)
        study_hours_per_day = st.number_input(
            "Study Hours Per Day", 0.0, 24.0, 4.0
        )
        sleep_hours = st.number_input(
            "Sleep Hours", 0.0, 24.0, 7.0
        )

    st.header("2. Technical Skills & Experience")

    col1, col2, col3 = st.columns(3)

    with col1:
        internships_count = st.number_input("Internships", 0, 20, 0)
        projects_count = st.number_input("Projects", 0, 50, 1)
        certifications_count = st.number_input(
            "Certifications", 0, 50, 0
        )
        coding_skill_score = st.slider(
            "Coding Skill Score", 0, 100, 50
        )
        github_repos = st.number_input("GitHub Repositories", 0, 500, 0)

    with col2:
        aptitude_score = st.slider("Aptitude Score", 0, 100, 50)
        communication_skill_score = st.slider(
            "Communication Skill Score", 0, 100, 50
        )
        logical_reasoning_score = st.slider(
            "Logical Reasoning Score", 0, 100, 50
        )
        hackathons_participated = st.number_input(
            "Hackathons Participated", 0, 50, 0
        )
        mock_interview_score = st.slider(
            "Mock Interview Score", 0, 100, 50
        )

    with col3:
        linkedin_connections = st.number_input(
            "LinkedIn Connections", 0, 10000, 100
        )
        extracurricular_score = st.slider(
            "Extracurricular Score", 0, 100, 50
        )
        leadership_score = st.slider(
            "Leadership Score", 0, 100, 50
        )
        volunteer_experience = st.selectbox(
            "Volunteer Experience", ["Yes", "No"]
        )

    submitted = st.form_submit_button(
        "🔍 Predict Placement",
        use_container_width=True
    )

if submitted:

    # Prepare student data in the exact training column order
    student_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "cgpa": cgpa,
        "branch": branch,
        "college_tier": college_tier,
        "internships_count": internships_count,
        "projects_count": projects_count,
        "certifications_count": certifications_count,
        "coding_skill_score": coding_skill_score,
        "aptitude_score": aptitude_score,
        "communication_skill_score": communication_skill_score,
        "logical_reasoning_score": logical_reasoning_score,
        "hackathons_participated": hackathons_participated,
        "github_repos": github_repos,
        "linkedin_connections": linkedin_connections,
        "mock_interview_score": mock_interview_score,
        "attendance_percentage": attendance_percentage,
        "backlogs": backlogs,
        "extracurricular_score": extracurricular_score,
        "leadership_score": leadership_score,
        "volunteer_experience": volunteer_experience,
        "sleep_hours": sleep_hours,
        "study_hours_per_day": study_hours_per_day
    }])

    # Preprocess and predict
    processed_data = preprocessor.transform(student_data)

    prediction = xgb_model.predict(processed_data)[0]
    probabilities = xgb_model.predict_proba(processed_data)[0]

    # Map probabilities using model class labels
    probability_map = {
        int(cls): float(prob)
        for cls, prob in zip(xgb_model.classes_, probabilities)
    }

    placed_probability = probability_map.get(1, 0) * 100
    not_placed_probability = probability_map.get(0, 0) * 100

    st.divider()
    st.header("3. Placement Prediction Result")

    if int(prediction) == 1:
        st.success(" Predicted Status: PLACED")
    else:
        st.error("Predicted Status: NOT PLACED")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Placed Probability",
            f"{placed_probability:.2f}%"
        )

    with col2:
        st.metric(
            "Not Placed Probability",
            f"{not_placed_probability:.2f}%"
        )

    st.caption(
        "Note: These are model-estimated probabilities, "
        "not a guarantee of actual placement."
    )
