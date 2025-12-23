import streamlit as st
import numpy as np
import joblib

# ---------------- Page configuration ----------------
st.set_page_config(
    page_title="Breast Cancer Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

# ---------------- Custom CSS ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #edf2ff 0%, #f8fafc 40%, #ffffff 100%);
    }
    .main {
        background-color: transparent;
    }
    .result-box {
        padding: 18px 20px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        background-color: #ffffffee;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
    }
    .title-text {
        font-size: 32px;
        font-weight: 700;
        color: #111827;
    }
    .subtitle-text {
        font-size: 14px;
        color: #4b5563;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Load model and scaler ----------------
# Make sure scaler.pkl and breast_cancer_model.pkl are in same folder as this file
scaler = joblib.load("scaler.pkl")
model = joblib.load("breast_cancer_model.pkl")

# ---------------- Header ----------------
left_col, right_col = st.columns([1, 4])

with left_col:
    # Put a logo file (e.g. logo.png) in the same folder, or comment this line out
    try:
        st.image("logo.png", width=70)
    except Exception:
        st.write("")  # no logo available

with right_col:
    st.markdown('<div class="title-text">Breast Cancer Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle-text">'
        'Interactive educational tool that estimates the probability of a tumor being malignant '
        'based on diagnostic measurements. Not a substitute for professional medical advice.'
        '</div>',
        unsafe_allow_html=True,
    )

st.write("")  # spacer

# ---------------- Input section ----------------
st.markdown("### Patient Measurements")

st.markdown(
    "Enter the cytology measurements below. "
    "You can start with the most important mean features and optionally add others."
)

# NOTE: These names must match the columns used when training your model and be in the same order.
# Here is an example using common key features. Extend this list if your model expects more.
with st.expander("Mean features (recommended)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        radius_mean = st.number_input("Radius (mean)", min_value=0.0, value=14.0, step=0.1)
        texture_mean = st.number_input("Texture (mean)", min_value=0.0, value=19.0, step=0.1)
        perimeter_mean = st.number_input("Perimeter (mean)", min_value=0.0, value=90.0, step=0.1)
        area_mean = st.number_input("Area (mean)", min_value=0.0, value=600.0, step=1.0)
    with c2:
        smoothness_mean = st.number_input(
            "Smoothness (mean)", min_value=0.0, value=0.10, step=0.001, format="%.3f"
        )
        compactness_mean = st.number_input(
            "Compactness (mean)", min_value=0.0, value=0.15, step=0.001, format="%.3f"
        )
        concavity_mean = st.number_input(
            "Concavity (mean)", min_value=0.0, value=0.20, step=0.001, format="%.3f"
        )
        concave_points_mean = st.number_input(
            "Concave points (mean)", min_value=0.0, value=0.10, step=0.001, format="%.3f"
        )

# If your final model uses more features (like *_se or *_worst), add another expander and inputs here
# and append them to feature_values in the same order they appeared in training.

# ---------------- Build feature vector ----------------
# ORDER MATTERS: this must match X.columns used to train the model.
feature_values = [
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    compactness_mean,
    concavity_mean,
    concave_points_mean,
]

st.write("")
run_button = st.button("🔍 Run prediction", use_container_width=True)

# ---------------- Prediction and output ----------------
if run_button:
    X = np.array(feature_values).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    proba_malignant = float(model.predict_proba(X_scaled)[0, 1])

    st.write("")
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    if pred == 1:
        st.markdown("### 🔴 Result: **Malignant (High Risk)**")
        st.markdown(f"Estimated probability of malignancy: **{proba_malignant:.2%}**")
        st.info(
            "This model suggests a high risk of malignancy. "
            "Clinical evaluation and additional tests are essential."
        )
    else:
        st.markdown("### 🟢 Result: **Benign (Lower Risk)**")
        st.markdown(f"Estimated probability of malignancy: **{proba_malignant:.2%}**")
        st.success(
            "Model indicates a lower probability of malignancy. "
            "Regular screening and professional medical advice are still important."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Model details and footer ----------------
with st.expander("ℹ️ Model details"):
    st.write("- Algorithm: Support Vector Machine (SVM) classifier")
    st.write("- Dataset: Breast Cancer Wisconsin (Diagnostic)")
    st.write("- Metrics (test set): around 98% accuracy with high ROC‑AUC")
    st.write("- Purpose: Learning project for machine learning and deployment practice")

st.markdown(
    "<div class='footer-text'>Built as an educational project • Not for real medical diagnosis</div>",
    unsafe_allow_html=True,
)
