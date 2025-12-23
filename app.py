import streamlit as st
import numpy as np
import joblib

# ================= Page configuration =================
st.set_page_config(
    page_title="Breast Cancer Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# ================= Global CSS =================
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 0% 0%, #e0ecff 0%, #f4f6fb 40%, #ffffff 100%);
        color: #0f172a;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .main { background-color: transparent; }

    .app-title {
        font-size: 26px;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #0f172a;
    }
    .app-subtitle {
        font-size: 13px;
        color: #6b7280;
        margin-top: 2px;
    }
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 500;
        background: #e0f2fe;
        color: #0369a1;
        margin-top: 6px;
    }

    .card {
        padding: 18px 20px;
        border-radius: 16px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 12px 30px rgba(15,23,42,0.06);
    }
    .card-soft {
        padding: 14px 16px;
        border-radius: 14px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
    }

    .section-title {
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 4px;
        color: #0f172a;
    }
    .section-caption {
        font-size: 12px;
        color: #6b7280;
        margin-bottom: 10px;
    }

    .result-benign {
        background: linear-gradient(135deg, #ecfdf5, #dcfce7);
        border-color: #16a34a;
    }
    .result-malignant {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border-color: #dc2626;
    }

    .footer-text {
        text-align: center;
        color: #9ca3af;
        font-size: 11px;
        margin-top: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================= Load model & scaler =================
scaler = joblib.load("scaler.pkl")
model = joblib.load("breast_cancer_model.pkl")

# ================= Top bar =================
with st.container():
    col_logo, col_text = st.columns([0.8, 4])

    with col_logo:
        try:
            st.image("logo.png", width=50)
        except Exception:
            st.write("")

    with col_text:
        st.markdown('<div class="app-title">Breast Cancer Risk Predictor</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="app-subtitle">'
            'Estimate malignancy risk from diagnostic measurements with a clean, modern interface. '
            'Built for learning; not intended for real medical decisions.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="pill">Educational demo · SVM model</div>', unsafe_allow_html=True)

st.markdown("---")

# ================= Tabs =================
tab_pred, tab_model, tab_help = st.tabs(["🔍 Prediction", "📊 Model", "ℹ️ Help"])

# ----------------------------------------------------------------------
# TAB 1: PREDICTION
# ----------------------------------------------------------------------
with tab_pred:
    col_left, col_right = st.columns([1.4, 1])

    # -------- LEFT: INPUTS --------
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patient measurements</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">'
            'Use the controls below to enter tumor characteristics. '
            'You can also load example cases from the dropdown.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Preset examples
        preset = st.selectbox(
            "Load example case (optional)",
            ["None", "Typical benign", "Typical malignant"],
            index=0,
        )

        # Base defaults (approx mid‑range)
        defaults = {
            "radius_mean": 14.0,
            "texture_mean": 19.0,
            "perimeter_mean": 90.0,
            "area_mean": 600.0,
            "smoothness_mean": 0.10,
            "compactness_mean": 0.15,
            "concavity_mean": 0.20,
            "concave points_mean": 0.10,
            "symmetry_mean": 0.19,
            "fractal_dimension_mean": 0.06,
            "radius_se": 0.4,
            "texture_se": 1.2,
            "perimeter_se": 3.0,
            "area_se": 40.0,
            "smoothness_se": 0.005,
            "compactness_se": 0.02,
            "concavity_se": 0.03,
            "concave points_se": 0.01,
            "symmetry_se": 0.02,
            "fractal_dimension_se": 0.003,
            "radius_worst": 16.0,
            "texture_worst": 25.0,
            "perimeter_worst": 105.0,
            "area_worst": 800.0,
            "smoothness_worst": 0.13,
            "compactness_worst": 0.25,
            "concavity_worst": 0.30,
            "concave points_worst": 0.15,
            "symmetry_worst": 0.30,
            "fractal_dimension_worst": 0.08,
        }

        if preset == "Typical benign":
            defaults.update({
                "radius_mean": 12.0,
                "texture_mean": 17.0,
                "perimeter_mean": 78.0,
                "area_mean": 470.0,
                "smoothness_mean": 0.09,
                "compactness_mean": 0.10,
                "concavity_mean": 0.08,
                "concave points_mean": 0.05,
                "radius_worst": 14.0,
                "area_worst": 600.0,
            })
        elif preset == "Typical malignant":
            defaults.update({
                "radius_mean": 18.0,
                "texture_mean": 22.0,
                "perimeter_mean": 120.0,
                "area_mean": 1000.0,
                "smoothness_mean": 0.11,
                "compactness_mean": 0.20,
                "concavity_mean": 0.25,
                "concave points_mean": 0.15,
                "radius_worst": 22.0,
                "area_worst": 1500.0,
            })

        # -------- Mean features --------
        with st.expander("Core mean features", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                radius_mean = st.number_input(
                    "Radius (mean)", 0.0, 40.0, defaults["radius_mean"], 0.1
                )
                texture_mean = st.number_input(
                    "Texture (mean)", 0.0, 40.0, defaults["texture_mean"], 0.1
                )
                perimeter_mean = st.number_input(
                    "Perimeter (mean)", 0.0, 200.0, defaults["perimeter_mean"], 0.1
                )
                area_mean = st.number_input(
                    "Area (mean)", 0.0, 3000.0, defaults["area_mean"], 1.0
                )
            with c2:
                smoothness_mean = st.number_input(
                    "Smoothness (mean)", 0.0, 1.0, defaults["smoothness_mean"], 0.001, format="%.3f"
                )
                compactness_mean = st.number_input(
                    "Compactness (mean)", 0.0, 1.0, defaults["compactness_mean"], 0.001, format="%.3f"
                )
                concavity_mean = st.number_input(
                    "Concavity (mean)", 0.0, 1.0, defaults["concavity_mean"], 0.001, format="%.3f"
                )
                concave_points_mean = st.number_input(
                    "Concave points (mean)", 0.0, 1.0, defaults["concave points_mean"], 0.001, format="%.3f"
                )

        with st.expander("Additional mean features", expanded=False):
            symmetry_mean = st.number_input(
                "Symmetry (mean)", 0.0, 1.0, defaults["symmetry_mean"], 0.001, format="%.3f"
            )
            fractal_dimension_mean = st.number_input(
                "Fractal dimension (mean)", 0.0, 1.0, defaults["fractal_dimension_mean"], 0.001, format="%.3f"
            )

        # -------- SE features --------
        with st.expander("Standard error (SE) features", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                radius_se = st.number_input(
                    "Radius SE", 0.0, 5.0, defaults["radius_se"], 0.01
                )
                texture_se = st.number_input(
                    "Texture SE", 0.0, 5.0, defaults["texture_se"], 0.01
                )
                perimeter_se = st.number_input(
                    "Perimeter SE", 0.0, 20.0, defaults["perimeter_se"], 0.1
                )
                area_se = st.number_input(
                    "Area SE", 0.0, 200.0, defaults["area_se"], 0.5
                )
            with c2:
                smoothness_se = st.number_input(
                    "Smoothness SE", 0.0, 0.1, defaults["smoothness_se"], 0.001, format="%.3f"
                )
                compactness_se = st.number_input(
                    "Compactness SE", 0.0, 0.1, defaults["compactness_se"], 0.001, format="%.3f"
                )
                concavity_se = st.number_input(
                    "Concavity SE", 0.0, 0.1, defaults["concavity_se"], 0.001, format="%.3f"
                )
                concave_points_se = st.number_input(
                    "Concave points SE", 0.0, 0.1, defaults["concave points_se"], 0.001, format="%.3f"
                )
            symmetry_se = st.number_input(
                "Symmetry SE", 0.0, 0.2, defaults["symmetry_se"], 0.001, format="%.3f"
            )
            fractal_dimension_se = st.number_input(
                "Fractal dimension SE", 0.0, 0.05, defaults["fractal_dimension_se"], 0.001, format="%.3f"
            )

        # -------- Worst features --------
        with st.expander("Worst features", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                radius_worst = st.number_input(
                    "Radius (worst)", 0.0, 50.0, defaults["radius_worst"], 0.1
                )
                texture_worst = st.number_input(
                    "Texture (worst)", 0.0, 50.0, defaults["texture_worst"], 0.1
                )
                perimeter_worst = st.number_input(
                    "Perimeter (worst)", 0.0, 300.0, defaults["perimeter_worst"], 0.1
                )
                area_worst = st.number_input(
                    "Area (worst)", 0.0, 4000.0, defaults["area_worst"], 1.0
                )
            with c2:
                smoothness_worst = st.number_input(
                    "Smoothness (worst)", 0.0, 1.0, defaults["smoothness_worst"], 0.001, format="%.3f"
                )
                compactness_worst = st.number_input(
                    "Compactness (worst)", 0.0, 1.0, defaults["compactness_worst"], 0.001, format="%.3f"
                )
                concavity_worst = st.number_input(
                    "Concavity (worst)", 0.0, 1.0, defaults["concavity_worst"], 0.001, format="%.3f"
                )
                concave_points_worst = st.number_input(
                    "Concave points (worst)", 0.0, 1.0, defaults["concave points_worst"], 0.001, format="%.3f"
                )
            symmetry_worst = st.number_input(
                "Symmetry (worst)", 0.0, 1.0, defaults["symmetry_worst"], 0.001, format="%.3f"
            )
            fractal_dimension_worst = st.number_input(
                "Fractal dimension (worst)", 0.0, 1.0, defaults["fractal_dimension_worst"], 0.001, format="%.3f"
            )

        # ---- Build 30-feature vector in the SAME ORDER as training ----
        feature_values = [
            radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
            compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
            radius_se, texture_se, perimeter_se, area_se, smoothness_se,
            compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
            radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
            compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst,
        ]

        run_button = st.button("Run prediction", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------- RIGHT: RESULT & SNAPSHOT --------
    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">'
            'The model outputs a class (benign / malignant) and an estimated probability of malignancy.'
            '</div>',
            unsafe_allow_html=True,
        )

        if run_button:
            X = np.array(feature_values).reshape(1, -1)
            X_scaled = scaler.transform(X)
            pred = int(model.predict(X_scaled)[0])
            proba_malignant = float(model.predict_proba(X_scaled)[0, 1])
            prob_percent = f"{proba_malignant * 100:.1f}%"

            if pred == 1:
                result_class = "result-malignant"
                heading = "Malignant (higher risk)"
                icon = "🔴"
            else:
                result_class = "result-benign"
                heading = "Benign (lower risk)"
                icon = "🟢"

            st.markdown(f'<div class="card-soft {result_class}">', unsafe_allow_html=True)
            st.markdown(f"**{icon} {heading}**")
            st.write(f"Estimated probability of malignancy: **{prob_percent}**")

            st.progress(min(max(proba_malignant, 0.0), 1.0))

            if pred == 1:
                st.info(
                    "The model suggests a higher probability of malignancy. "
                    "In real scenarios, imaging, biopsy, and specialist review would be required."
                )
            else:
                st.success(
                    "The model suggests a lower probability of malignancy. "
                    "In real practice, screening schedules and clinician judgement still matter."
                )

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Fill in the measurements on the left and click **Run prediction** to see results.")

        st.write("")
        st.markdown('<div class="card-soft">', unsafe_allow_html=True)
        st.markdown("**Model snapshot**")
        st.write("- Type: Support Vector Machine (SVM) classifier")
        st.write("- Dataset: Breast Cancer Wisconsin (Diagnostic)")
        st.write("- Features: 30 numeric descriptors (mean, SE, worst values)")
        st.write("- Performance: ~98% accuracy on held‑out test data")
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# TAB 2: MODEL DETAILS
# ----------------------------------------------------------------------
with tab_model:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Model details")
    st.write("- Algorithm: SVM with RBF kernel, tuned hyperparameters.")
    st.write("- Training: 80/20 train–test split with stratification.")
    st.write("- Feature groups: mean, standard error, and worst‑case measurements.")
    st.write("- Strengths: high accuracy and strong separation between classes.")
    st.write("- Limitations: trained on one dataset; not calibrated for clinical deployment.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# TAB 3: HELP / INTERPRETATION
# ----------------------------------------------------------------------
with tab_help:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ How to interpret this tool")
    st.write("- Built for **learning end‑to‑end ML** (EDA → model → deployment).")
    st.write("- Not approved or validated for real‑world diagnosis or treatment.")
    st.write("- Use predictions to understand how features affect model output, not to make clinical decisions.")
    st.write("- Combining this with your notebook and README makes a strong portfolio project.")
    st.markdown('</div>', unsafe_allow_html=True)

# ================= Footer =================
st.markdown(
    "<div class='footer-text'>Educational ML project · Not a medical device or diagnostic tool.</div>",
    unsafe_allow_html=True,
)
