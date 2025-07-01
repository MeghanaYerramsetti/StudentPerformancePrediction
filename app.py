import streamlit as st
import pandas as pd
import joblib

# Load trained models
ridge_model = joblib.load("ridge_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

st.title("🎓 Student Final Grade Predictor")
st.write("Enter the details below to predict your final exam grade (G3).")

# Input fields
studytime = st.slider("Study Time (1 to 4)", 1, 4, 2)
failures = st.slider("Past Class Failures", 0, 3, 0)
absences = st.slider("Total Absences", 0, 30, 5)
goout = st.slider("Going Out (1 = rarely, 5 = very often)", 1, 5, 3)
health = st.slider("Health (1 = poor, 5 = excellent)", 1, 5, 3)
G1 = st.slider("Grade in Period 1 (G1)", 0, 20, 10)
G2 = st.slider("Grade in Period 2 (G2)", 0, 20, 10)

if st.button("Predict Final Grade (G3)"):
    features = pd.DataFrame([[studytime, failures, absences, goout, health, G1, G2]],
                            columns=['studytime', 'failures', 'absences', 'goout', 'health', 'G1', 'G2'])

    ridge_pred = ridge_model.predict(features)[0]
    xgb_pred = xgb_model.predict(features)[0]

    st.write(f"📘 Ridge Regression Prediction: **{ridge_pred:.2f}**")
    st.write(f"📗 XGBoost Prediction: **{xgb_pred:.2f}**")

    if abs(ridge_pred - G2) < abs(xgb_pred - G2):
        st.success("✅ Ridge model is closer to your last score.")
    else:
        st.success("✅ XGBoost model seems more accurate.")
