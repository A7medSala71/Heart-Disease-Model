"""
Simple Streamlit UI for the trained Heart Disease classifier.
- Loads models/final_model.pkl (exported by main.py)
- Reindexes user input to match saved training columns (models/train_columns.json)
- Presents inputs for common Cleveland features (safe defaults)
- Returns predicted probability, class, and lets you change the decision threshold
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

st.set_page_config(page_title="Heart Disease Risk – UCI", layout="centered")

st.title("❤️ Heart Disease Risk Predictor (UCI)")
st.caption("Model = preprocessing + tuned classifier pipeline exported by `main.py`.")

MODEL_PATH = "models/final_model.pkl"
COLS_PATH = "models/train_columns.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(COLS_PATH):
    st.error("Artifacts not found. Please run `python main.py` first to train and export the model.")
    st.stop()

# Load model and training columns
model = joblib.load(MODEL_PATH)
with open(COLS_PATH, "r") as f:
    train_cols = json.load(f)["train_columns"]

st.subheader("Enter Patient Data")

# Common Cleveland attributes & sensible defaults.
# If your training columns differ, we will still reindex to match them.
input_spec = {
    "age": ("number", 55.0),
    "sex": ("select", [0, 1], 1),
    "cp": ("select", [0, 1, 2, 3], 0),             # chest pain type
    "trestbps": ("number", 130.0),                 # resting blood pressure
    "chol": ("number", 246.0),                     # serum cholesterol
    "fbs": ("select", [0, 1], 0),                  # fasting blood sugar > 120 mg/dl
    "restecg": ("select", [0, 1, 2], 1),           # resting ECG results
    "thalach": ("number", 150.0),                  # max heart rate achieved
    "exang": ("select", [0, 1], 0),                # exercise-induced angina
    "oldpeak": ("number", 1.0),                    # ST depression
    "slope": ("select", [0, 1, 2], 1),             # slope of peak exercise ST segment
    "ca": ("select", [0, 1, 2, 3, 4], 0),          # # major vessels colored by fluoroscopy
    "thal": ("select", [0, 1, 2, 3], 2)            # thalassemia (codes can vary)
}

cols = st.columns(2)
user_vals = {}
i = 0
for feat, spec in input_spec.items():
    with cols[i % 2]:
        if spec[0] == "number":
            user_vals[feat] = st.number_input(feat, value=float(spec[1]))
        elif spec[0] == "select":
            choices = spec[1]; default = spec[2]
            user_vals[feat] = st.selectbox(feat, choices, index=choices.index(default))
    i += 1

# Optional: users can add any additional key:value pairs to see how the pipeline handles them
with st.expander("Optional: Add custom fields (advanced)"):
    extra_raw = st.text_area("Enter JSON (e.g., {\"some_new_col\": 1})", value="{}")
    try:
        extra = json.loads(extra_raw) if extra_raw.strip() else {}
    except Exception:
        extra = {}
        st.warning("Invalid JSON; ignoring.")

# Build a single-row DataFrame
X_user = pd.DataFrame([{**user_vals, **extra}])

# Reindex to match training columns (missing columns become NaN; extras are dropped)
X_user = X_user.reindex(columns=train_cols)

st.markdown("---")
threshold = st.slider("Decision threshold (class = 1 if proba ≥ threshold)", 0.05, 0.95, 0.50, 0.01)

if st.button("Predict"):
    try:
        proba = float(model.predict_proba(X_user)[:, 1][0])
        pred = int(proba >= threshold)
        st.success(f"Predicted Probability of Heart Disease: {proba:.3f}")
        st.write(f"Predicted Class @ threshold {threshold:.2f}: **{pred}** (1 = Disease, 0 = No Disease)")
        with st.expander("Show input sent to the model"):
            st.dataframe(X_user)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("If you retrained with different columns, adjust/add fields or retrain via `python main.py`.")
