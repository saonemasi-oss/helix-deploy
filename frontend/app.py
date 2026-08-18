"""
HELIX Phase 6 Streamlit Dashboard
Calls the FastAPI backend and presents the four models + ontology.

The backend derives engineered features (BMI, HRR, training-intensity index,
lean body mass, calorie ratios, frequency x experience) from raw measurements,
and returns HTTP 422 naming any field it cannot derive. This client therefore
sends raw measurements only, and surfaces backend errors rather than crashing.
"""
import streamlit as st
import requests
import pandas as pd

API = "https://helix-backend-150551383910.europe-west2.run.app"
TIMEOUT = 60

st.set_page_config(page_title="HELIX", page_icon="H", layout="wide")
st.title("HELIX - Muscle Growth Prediction & Training Platform")
st.caption("MSc Data Science Project - QMUL - Sawan Masih Nayak")


# ---------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------
def call_api(method, path, **kw):
    """Return (ok, payload). Never raises; shows a readable message instead."""
    try:
        r = requests.request(method, f"{API}{path}", timeout=TIMEOUT, **kw)
    except Exception as exc:
        st.error(f"Could not reach the backend: {exc}")
        return False, None

    if r.status_code == 200:
        return True, r.json()

    # The backend names missing fields explicitly - show that, not a traceback.
    try:
        detail = r.json().get("detail", {})
    except Exception:
        detail = {}
    if isinstance(detail, dict) and detail.get("missing"):
        st.warning(
            "The model needs values this form did not supply: "
            + ", ".join(detail["missing"])
        )
    else:
        st.error(f"Backend returned HTTP {r.status_code}: {r.text[:300]}")
    return False, None


def age_group(age):
    """One-hot age band. 18-25 is the reference level and has no column."""
    return {
        "Age_Group_26-35": int(26 <= age <= 35),
        "Age_Group_36-45": int(36 <= age <= 45),
        "Age_Group_46-55": int(46 <= age <= 55),
        "Age_Group_56+":   int(age >= 56),
    }


def bmi_category(bmi):
    """One-hot BMI band. Normal weight is the reference level."""
    return {
        "BMI_Category_Underweight": int(bmi < 18.5),
        "BMI_Category_Overweight":  int(25.0 <= bmi < 30.0),
        "BMI_Category_Obese":       int(bmi >= 30.0),
    }


# Confirm backend is alive
ok, status = call_api("GET", "/")
if not ok:
    st.stop()
st.success(
    f"Backend connected - {status['ontology_triples']} ontology triples, "
    f"{len(status['models'])} models loaded"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Body Composition", "Archetype", "Recovery Forecast", "Training Philosophy"])

# ---------------------------------------------------------------
#  TAB 1: Body fat + calories
# ---------------------------------------------------------------
with tab1:
    st.header("Body Composition & Energy Prediction")
    st.caption("Enter raw measurements. BMI, heart-rate reserve, the "
               "training-intensity index and the calorie ratios are derived "
               "by the backend.")
    c1, c2, c3 = st.columns(3)
    with c1:
        age      = st.number_input("Age", 18, 80, 28)
        weight   = st.number_input("Weight (kg)", 40.0, 160.0, 82.0)
        height   = st.number_input("Height (m)", 1.40, 2.20, 1.78)
        gender_m = st.selectbox("Gender", ["Male", "Female"]) == "Male"
    with c2:
        max_bpm  = st.number_input("Max BPM", 120, 220, 185)
        avg_bpm  = st.number_input("Avg session BPM", 80, 200, 128)
        rest_bpm = st.number_input("Resting BPM", 35, 110, 62)
        water    = st.number_input("Water intake (L)", 0.5, 5.0, 2.5)
    with c3:
        session  = st.number_input("Session duration (hours)", 0.25, 3.0, 1.30)
        calories = st.number_input("Calories burned this session", 50.0, 2000.0, 900.0)
        fat_pct  = st.number_input("Current body fat (%)", 3.0, 60.0, 24.0)
        freq     = st.number_input("Workout frequency (days/week)", 1, 7, 4)
        exp      = st.selectbox("Experience level", [1, 2, 3],
                                format_func=lambda x: {1: "Beginner",
                                                       2: "Intermediate",
                                                       3: "Advanced"}[x])

    if st.button("Predict body composition"):
        feats = {
            "Age": age, "Weight_kg": weight, "Height_m": height,
            "Max_BPM": max_bpm, "Avg_BPM": avg_bpm, "Resting_BPM": rest_bpm,
            "Session_Duration_hours": session,
            "Calories_Burned": calories,
            "Fat_Percentage": fat_pct,
            "Water_Intake_liters": water,
            "Workout_Frequency_days_per_week": freq,
            "Experience_Level": exp,
        }
        m1, m2 = st.columns(2)
        ok_bf, bf = call_api("POST", "/predict/bodyfat", json={"features": feats})
        if ok_bf:
            m1.metric("Predicted body fat", f"{bf['predicted_body_fat_pct']} %")
        ok_cal, cal = call_api("POST", "/predict/calories", json={"features": feats})
        if ok_cal:
            m2.metric("Predicted calories/session", f"{cal['predicted_calories']:.0f}")
        if ok_bf and ok_cal:
            st.caption("Body fat is predicted from the session's energy "
                       "expenditure; calories are predicted from body "
                       "composition. Each model uses the other quantity as an "
                       "input, which is why both are entered above.")

# ---------------------------------------------------------------
#  TAB 2: Archetype
# ---------------------------------------------------------------
with tab2:
    st.header("Athlete Archetype Classification")
    st.write("Classifies an athlete as a **Committed Athlete** or "
             "**General Population** member.")
    c1, c2, c3 = st.columns(3)
    with c1:
        b_age    = st.number_input("Age ", 18, 80, 28, key="b_age")
        b_weight = st.number_input("Weight (kg) ", 40.0, 160.0, 82.0, key="b_wt")
        b_height = st.number_input("Height (m) ", 1.40, 2.20, 1.78, key="b_ht")
        b_gender = st.selectbox("Gender ", ["Male", "Female"], key="b_gen") == "Male"
    with c2:
        b_max    = st.number_input("Max BPM ", 120, 220, 185, key="b_max")
        b_avg    = st.number_input("Avg session BPM ", 80, 200, 128, key="b_avg")
        b_rest   = st.number_input("Resting BPM ", 35, 110, 62, key="b_rest")
        b_water  = st.number_input("Water intake (L) ", 0.5, 5.0, 2.5, key="b_wat")
    with c3:
        b_sess   = st.number_input("Session duration (hours) ", 0.25, 3.0, 1.30, key="b_ses")
        b_cal    = st.number_input("Calories burned ", 50.0, 2000.0, 900.0, key="b_cal")
        b_fat    = st.number_input("Body fat (%) ", 3.0, 60.0, 24.0, key="b_fat")
        b_freq   = st.number_input("Workout frequency (days/week) ", 1, 7, 4, key="b_frq")
        b_exp    = st.selectbox("Experience level ", [1, 2, 3], index=1, key="b_exp",
                                format_func=lambda x: {1: "Beginner",
                                                       2: "Intermediate",
                                                       3: "Advanced"}[x])
    b_type   = st.selectbox("Workout type",
                            ["Cardio", "HIIT", "Strength", "Yoga"], index=1)
    b_commit = st.selectbox("Training commitment",
                            ["Casual", "Regular", "Dedicated"], index=2)

    if st.button("Classify archetype"):
        b_bmi = b_weight / (b_height ** 2)
        feats = {
            "Age": b_age, "Weight_kg": b_weight, "Height_m": b_height,
            "Max_BPM": b_max, "Avg_BPM": b_avg, "Resting_BPM": b_rest,
            "Session_Duration_hours": b_sess,
            "Calories_Burned": b_cal,
            "Fat_Percentage": b_fat,
            "Water_Intake_liters": b_water,
            "Workout_Frequency_days_per_week": b_freq,
            "Experience_Level": b_exp,
            # categorical choices the backend cannot infer
            "Gender_Male": int(b_gender),
            "Workout_Type_HIIT":     int(b_type == "HIIT"),
            "Workout_Type_Strength": int(b_type == "Strength"),
            "Workout_Type_Yoga":     int(b_type == "Yoga"),
            "Training_Commitment_Dedicated": int(b_commit == "Dedicated"),
            "Training_Commitment_Regular":   int(b_commit == "Regular"),
        }
        feats.update(age_group(b_age))
        feats.update(bmi_category(b_bmi))
        ok_a, r = call_api("POST", "/predict/archetype", json={"features": feats})
        if ok_a:
            st.metric("Predicted archetype", r["predicted_archetype"])

# ---------------------------------------------------------------
#  TAB 3: Recovery LSTM
# ---------------------------------------------------------------
with tab3:
    st.header("Next-Day Recovery Readiness Forecast (LSTM)")
    st.write("Select a held-out test user. The LSTM forecasts each day's "
             "readiness from the previous 7 days, shown against the actual "
             "value. These users were absent from training entirely.")
    ok_u, users_payload = call_api("GET", "/lstm/users")
    if ok_u:
        uid = st.selectbox("Test user", users_payload["users"])
        if st.button("Forecast recovery"):
            ok_r, r = call_api("GET", f"/predict/recovery/{uid}")
            if ok_r:
                df = pd.DataFrame({
                    "date": pd.to_datetime(r["dates"]),
                    "Predicted": r["predicted"],
                    "Actual": r["actual"],
                }).set_index("date")
                st.line_chart(df)
                err = (df["Predicted"] - df["Actual"]).abs().mean()
                st.metric("Mean absolute error", f"{err:.1f} readiness points")

# ---------------------------------------------------------------
#  TAB 4: Ontology
# ---------------------------------------------------------------
with tab4:
    st.header("Training Philosophy Knowledge Graph")
    phil = st.selectbox("Philosophy",
                        ["HIT", "HighVolume", "ClassicalAesthetic", "ModernClassicPhil"])
    if st.button("Show principles"):
        ok_p, r = call_api("GET", f"/ontology/philosophy/{phil}")
        if ok_p:
            st.subheader(f"Principles of {phil}")
            for p in r["principles"]:
                st.write(f"- {p}")
            if not r["principles"]:
                st.info("No principles recorded for this philosophy.")