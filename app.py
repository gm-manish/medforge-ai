import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta

st.set_page_config(page_title="MedForge AI - Mini Cohort Generator")

st.title("🧬 MedForge AI - Mini Cohort Generator")
st.write("Generate HIPAA-safe synthetic patient timelines for an entire cohort.")

# --- UI ---
disease = st.selectbox("Select Disease Profile:", ["Hypertension", "Type 2 Diabetes", "Depression", "Heart Failure"])
n_patients = st.number_input("Number of Patients to Generate", min_value=1, max_value=100, value=10)

# --- Utility ---
def generate_patient_id():
    return f"PT-{random.randint(1000, 9999)}"

def get_random_ethnicity():
    return random.choice(["White", "Black", "Hispanic", "Asian", "Other"])

def get_random_gender():
    return random.choice(["M", "F"])

def get_base_date():
    return datetime.strptime("2025-04-01", "%Y-%m-%d")

# --- Disease Generators ---
def generate_hypertension_patient():
    pid = generate_patient_id()
    age = random.randint(45, 75)
    gender = get_random_gender()
    ethnicity = get_random_ethnicity()
    diagnosis = "I10"

    systolic = random.randint(150, 160)
    diastolic = random.randint(90, 100)
    hr = random.randint(75, 85)
    meds = []
    base_date = get_base_date()

    timeline = []
    for day in range(30):
        current_date = (base_date + timedelta(days=day)).strftime("%Y-%m-%d")
        note = "Follow-up."
        if day == 0:
            note = "Initial visit. BP elevated."
        elif day == 1:
            meds = ["Lisinopril"]
            note = "Started Lisinopril."

        systolic = max(120, systolic - random.randint(0, 2))
        diastolic = max(80, diastolic - random.randint(0, 1))
        hr = max(65, hr - random.randint(0, 1))

        timeline.append({
            "patient_id": pid,
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity,
            "diagnosis": diagnosis,
            "date": current_date,
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
            "heart_rate": hr,
            "meds_administered": ", ".join(meds),
            "notes": note
        })

    return timeline

def generate_diabetes_patient():
    pid = generate_patient_id()
    age = random.randint(35, 80)
    gender = get_random_gender()
    ethnicity = get_random_ethnicity()
    diagnosis = "E11"

    glucose = random.randint(180, 220)
    meds = []
    base_date = get_base_date()

    timeline = []
    for day in range(30):
        current_date = (base_date + timedelta(days=day)).strftime("%Y-%m-%d")
        note = "Follow-up."
        if day == 0:
            note = "Initial visit. Elevated glucose."
        elif day == 1:
            meds = ["Metformin"]
            note = "Started Metformin."

        glucose = max(90, glucose - random.randint(2, 5))

        timeline.append({
            "patient_id": pid,
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity,
            "diagnosis": diagnosis,
            "date": current_date,
            "glucose_mg_dl": glucose,
            "meds_administered": ", ".join(meds),
            "notes": note
        })

    return timeline

def generate_depression_patient():
    pid = generate_patient_id()
    age = random.randint(20, 65)
    gender = get_random_gender()
    ethnicity = get_random_ethnicity()
    diagnosis = "F33"

    mood_score = random.randint(2, 4)
    meds = []
    base_date = get_base_date()
    notes = [
        "Mood tracking.",
        "Patient reported low motivation.",
        "Slept better than previous night.",
        "Anxious about work.",
        "Stable mood today.",
        "Mild improvement noted.",
        "Reported sadness in the evening."
    ]

    timeline = []
    for day in range(30):
        current_date = (base_date + timedelta(days=day)).strftime("%Y-%m-%d")
        if day == 0:
            note = "Initial consult. Severe depressive symptoms."
        elif day == 1:
            meds = ["Sertraline"]
            note = "Started Sertraline."
        else:
            note = random.choice(notes)

        mood_score = max(1, min(5, mood_score + random.choice([-1, 0, 1])))

        timeline.append({
            "patient_id": pid,
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity,
            "diagnosis": diagnosis,
            "date": current_date,
            "mood_score": mood_score,
            "meds_administered": ", ".join(meds),
            "notes": note
        })

    return timeline

def generate_heart_failure_patient():
    pid = generate_patient_id()
    age = random.randint(50, 85)
    gender = get_random_gender()
    ethnicity = get_random_ethnicity()
    diagnosis = "I50"

    weight = random.uniform(90, 100)
    meds = []
    base_date = get_base_date()

    timeline = []
    for day in range(30):
        current_date = (base_date + timedelta(days=day)).strftime("%Y-%m-%d")
        if day == 0:
            note = "Initial visit. Weight gain and fatigue."
        elif day == 1:
            meds = ["Furosemide"]
            note = "Started Furosemide."
        else:
            note = "Ongoing weight monitoring."

        weight = max(65.0, weight - random.uniform(0.0, 0.4))

        timeline.append({
            "patient_id": pid,
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity,
            "diagnosis": diagnosis,
            "date": current_date,
            "weight_kg": round(weight, 1),
            "meds_administered": ", ".join(meds),
            "notes": note
        })

    return timeline

#---------------------------------
# Add Hypotension Prediction Tab
# ----------------------------------------

import pickle
from utils import preprocess_vitals  # <- we'll create utils.py with simple preprocessing

# Load your trained model
with open('icu_hypotension_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

# --- New Tabs ---
tab1, tab2 = st.tabs(["🧬 Generate Synthetic Cohort", "📤 Upload Vitals + Predict Hypotension"])

with tab1:

    # --- Main Generator Trigger ---
    if st.button("Generate Cohort"):
        all_patients = []

        for _ in range(n_patients):
            if disease == "Hypertension":
                all_patients.extend(generate_hypertension_patient())
            elif disease == "Type 2 Diabetes":
                all_patients.extend(generate_diabetes_patient())
            elif disease == "Depression":
                all_patients.extend(generate_depression_patient())
            elif disease == "Heart Failure":
                all_patients.extend(generate_heart_failure_patient())

        df = pd.DataFrame(all_patients)
        st.success(f"✅ Generated {n_patients} patient timelines.")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cohort CSV",
            data=csv,
            file_name=f"{disease.replace(' ', '_')}_synthetic_cohort.csv",
            mime="text/csv",
    )
    pass

with tab2:
    st.header("Upload ICU Vitals for Hypotension Prediction")
    uploaded_file = st.file_uploader("Upload Vitals CSV", type=["csv"])

    if uploaded_file:
        uploaded_data = pd.read_csv(uploaded_file)
        st.dataframe(uploaded_data)

        if st.button('🔮 Predict Hypotension'):
            X = preprocess_vitals(uploaded_data)
            preds = model.predict(X)
            uploaded_data['Hypotension Risk'] = preds

            st.success('✅ Prediction Done! Check Below:')
            st.dataframe(uploaded_data)

            # Highlight high-risk rows
            def highlight_risk(row):
                color = 'background-color: pink' if row['Hypotension Risk'] == 1 else ''
                return [color]*len(row)
            
            st.dataframe(uploaded_data.style.apply(highlight_risk, axis=1))

            st.download_button(
                "⬇️ Download Prediction CSV",
                uploaded_data.to_csv(index=False),
                "predictions.csv",
                "text/csv"
            )
