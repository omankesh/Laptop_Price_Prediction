import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Set correct path to parent directory
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load Models and Encoders
with open(os.path.join(base_path, "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

with open(os.path.join(base_path, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(base_path, "lightgbm_log_model.pkl"), "rb") as f:
    lgb_model = pickle.load(f)

with open(os.path.join(base_path, "gradient_boosting_log_model.pkl"), "rb") as f:
    gb_model = pickle.load(f)

with open(os.path.join(base_path, "xgboost_log_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)

st.title("💻 Laptop Price Predictor")

# --- Input fields ---
company = st.selectbox("Company", options=label_encoders["Company"].classes_)
typename = st.selectbox("Type", options=label_encoders["TypeName"].classes_)
ram = st.slider("RAM (in GB)", 2, 64, step=2)
weight = st.number_input("Weight (in Kg)", min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.selectbox("Touchscreen", options=["No", "Yes"])
ips = st.selectbox("IPS Display", options=["No", "Yes"])
screen_size = st.number_input("Screen Size (in inches)", 10.0, 18.0, step=0.1)
resolution = st.selectbox("Screen Resolution", ["1920x1080", "1366x768", "1600x900", "3840x2160", "3200x1800", "2880x1800"])

cpu = st.selectbox("CPU Type", options=label_encoders["Cpu_type"].classes_)
hdd = st.slider("HDD (in GB)", 0, 2000, step=128)
ssd = st.slider("SSD (in GB)", 0, 2000, step=128)

gpu = st.selectbox("GPU Type", options=label_encoders["Gpu_type"].classes_)
os = st.selectbox("Operating System", options=label_encoders["OS"].classes_)

if st.button("💰 Predict Laptop Price"):
    # --- Feature Engineering ---
    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    # Extract resolution
    res_width, res_height = map(int, resolution.split('x'))
    ppi = ((res_width**2 + res_height**2)**0.5) / screen_size

    # Apply label encoding
    company_enc = label_encoders["Company"].transform([company])[0]
    typename_enc = label_encoders["TypeName"].transform([typename])[0]
    cpu_enc = label_encoders["Cpu_type"].transform([cpu])[0]
    gpu_enc = label_encoders["Gpu_type"].transform([gpu])[0]
    os_enc = label_encoders["OS"].transform([os])[0]

    # Final input features (only numeric for scaler)
    input_features = np.array([[ram, weight, ppi, hdd, ssd]])
    scaled_input = scaler.transform(input_features)

    # Model predictions (Log values)
    log_pred_lgb = lgb_model.predict(scaled_input)[0]
    log_pred_gb = gb_model.predict(scaled_input)[0]
    log_pred_xgb = xgb_model.predict(scaled_input)[0]

    # Weighted average (can tune weights later)
    final_log_pred = (log_pred_lgb + log_pred_gb + log_pred_xgb) / 3
    final_price = np.exp(final_log_pred)

    st.success(f"🤑 Predicted Laptop Price: ₹ {int(final_price):,}")
