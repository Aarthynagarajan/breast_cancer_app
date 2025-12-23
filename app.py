import streamlit as st
import numpy as np
import joblib

# load saved objects
scaler = joblib.load("scaler.pkl")
model = joblib.load("breast_cancer_model.pkl")

st.title("Breast Cancer Classification App")

feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean",
    "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave points_mean"
]

inputs = []
for name in feature_names:
    value = st.number_input(name, value=0.0)
    inputs.append(value)

if st.button("Predict"):
    x = np.array(inputs).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    proba = model.predict_proba(x_scaled)[0, 1]

    if pred == 1:
        st.error(f"Prediction: Malignant (1), probability = {proba:.3f}")
    else:
        st.success(f"Prediction: Benign (0), probability = {proba:.3f}")
