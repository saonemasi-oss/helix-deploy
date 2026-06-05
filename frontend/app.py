"""
HELIX Phase 6 Streamlit Dashboard
Calls the FastAPI backend and presents the four models + ontology.
"""
import streamlit as st
import requests
import pandas as pd

API = "https://helix-backend-150551383910.europe-west2.run.app"

st.set_page_config(page_title="HELIX", page_icon=" ", layout="wide")
st.title("HELIX - Muscle Growth Prediction & Training Platform")
st.caption("MSc Data Science Project - QMUL - Sawan Masih Nayak")

# Confirm backend is alive
try:
    status = requests.get(f"{API}/", timeout=5).json()
    st.success(f" Backend connected {status['ontology_triples']} ontology triples "
               f"{len(status['models'])} models loaded")
except Exception as e:
    st.error(f" Cannot reach backend at {API}. Is uvicorn running? ({e})")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    [" Body Composition", " Archetype", " Recovery Forecast", " Training Philosophy"])

# ?????? TAB 1: Body fat + calories ??????
with tab1:
    st.header("Body Composition & Energy Prediction")
    c1, c2 = st.columns(2)
    with c1:
        age      = st.number_input("Age", 18, 80, 30)
        weight   = st.number_input("Weight (kg)", 40.0, 160.0, 75.0)
        height   = st.number_input("Height (m)", 1.4, 2.2, 1.75)
        avg_bpm  = st.number_input("Avg session BPM", 80, 200, 140)
    with c2:
        session  = st.number_input("Session duration (hours)", 0.25, 3.0, 1.0)
        water    = st.number_input("Water intake (L)", 0.5, 5.0, 2.5)
        freq     = st.number_input("Workout frequency (days/week)", 1, 7, 4)
        exp      = st.selectbox("Experience level", [1, 2, 3],
                                format_func=lambda x: {1:"Beginner",2:"Intermediate",3:"Advanced"}[x])

    if st.button("Predict body composition"):
        feats = {
            "Age": age, "Weight_kg": weight, "Height_m": height,
            "Avg_BPM": avg_bpm, "Session_Duration_hours": session,
            "Water_Intake_liters": water,
            "Workout_Frequency_days_per_week": freq, "Experience_Level": exp,
            "BMI": round(weight / (height**2), 1),
        }
        bf  = requests.post(f"{API}/predict/bodyfat",  json={"features": feats}).json()
        cal = requests.post(f"{API}/predict/calories", json={"features": feats}).json()
        m1, m2 = st.columns(2)
        m1.metric("Predicted body fat", f"{bf['predicted_body_fat_pct']} %")
        m2.metric("Predicted calories/session", f"{cal['predicted_calories']:.0f}")

#  TAB 2: Archetype 
with tab2:
    st.header("Athlete Archetype Classification")
    st.write("Classifies an athlete as a **Committed Athlete** or **General Population** member.")
    c1, c2 = st.columns(2)
    with c1:
        a_freq = st.slider("Workout frequency (days/week)", 1, 7, 4)
        a_exp  = st.slider("Experience level", 1, 3, 2)
    with c2:
        a_sess = st.slider("Session duration (hours)", 0.25, 3.0, 1.0)
        a_fat  = st.slider("Body fat %", 5.0, 40.0, 20.0)
    if st.button("Classify archetype"):
        feats = {"Workout_Frequency_days_per_week": a_freq, "Experience_Level": a_exp,
                 "Session_Duration_hours": a_sess, "Fat_Percentage": a_fat,
                 "Freq_x_Exp": a_freq * a_exp}
        r = requests.post(f"{API}/predict/archetype", json={"features": feats}).json()
        st.metric("Predicted archetype", r["predicted_archetype"])

# ?????? TAB 3: Recovery LSTM ??????
with tab3:
    st.header("Next-Day Recovery Readiness Forecast (LSTM)")
    st.write("Select a held-out test user. The LSTM forecasts each day's readiness "
             "from the previous 7 days ??? shown against the actual value.")
    users = requests.get(f"{API}/lstm/users").json()["users"]
    uid = st.selectbox("Test user", users)
    if st.button("Forecast recovery"):
        r = requests.get(f"{API}/predict/recovery/{uid}").json()
        df = pd.DataFrame({"date": pd.to_datetime(r["dates"]),
                           "Predicted": r["predicted"], "Actual": r["actual"]}).set_index("date")
        st.line_chart(df)
        err = (df["Predicted"] - df["Actual"]).abs().mean()
        st.metric("Mean absolute error", f"{err:.1f} readiness points")

# ?????? TAB 4: Ontology ??????
with tab4:
    st.header("Training Philosophy Knowledge Graph")
    phil = st.selectbox("Philosophy",
                        ["HIT", "HighVolume", "ClassicalAesthetic", "ModernClassicPhil"])
    if st.button("Show principles"):
        r = requests.get(f"{API}/ontology/philosophy/{phil}").json()
        st.subheader(f"Principles of {phil}")
        for p in r["principles"]:
            st.write(f"- {p}")
        if not r["principles"]:
            st.info("No principles recorded for this philosophy.")
