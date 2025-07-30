import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load models and other components
@st.cache_resource
def load_models():
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("lightgbm_log_model.pkl", "rb") as f:
        lgbm_model = pickle.load(f)
    with open("xgboost_log_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    return label_encoders, scaler, lgbm_model, xgb_model

label_encoders, scaler, lgbm_model, xgb_model = load_models()

# App Title
st.set_page_config(page_title="💻 Laptop Price Predictor", layout="centered")
st.title("💻 Laptop Price Predictor")
st.markdown("Enter laptop details below to predict its approximate price.")

# Sidebar Inputs
company = st.selectbox("Brand", options=label_encoders["Company"].classes_)
typename = st.selectbox("Type", options=label_encoders["TypeName"].classes_)
ram = st.selectbox("RAM (GB)", options=[4, 8, 16, 32, 64])
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
touchscreen = st.selectbox("Touchscreen", options=["No", "Yes"])
ips = st.selectbox("IPS Display", options=["No", "Yes"])
screen_size = st.number_input("Screen Size (inches)", min_value=10.0, max_value=18.0, value=15.6, step=0.1)
resolution = st.selectbox("Screen Resolution", options=[
    "1920x1080", "1366x768", "1600x900", "3840x2160", "3200x1800", "2880x1800", "2560x1600", "2560x1440"
])
cpu = st.selectbox("CPU", options=label_encoders["Cpu brand"].classes_)
hdd = st.selectbox("HDD (GB)", options=[0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox("SSD (GB)", options=[0, 128, 256, 512, 1024, 2048])
gpu = st.selectbox("GPU Brand", options=label_encoders["Gpu brand"].classes_)
os = st.selectbox("Operating System", options=label_encoders["os"].classes_)

# Calculate PPI
def calculate_ppi(res, size):
    x_res, y_res = map(int, res.split('x'))
    return ((x_res ** 2 + y_res ** 2) ** 0.5) / size

ppi = calculate_ppi(resolution, screen_size)

# Encode inputs
input_dict = {
    'Company': label_encoders["Company"].transform([company])[0],
    'TypeName': label_encoders["TypeName"].transform([typename])[0],
    'Ram': ram,
    'Weight': weight,
    'Touchscreen': 1 if touchscreen == "Yes" else 0,
    'Ips': 1 if ips == "Yes" else 0,
    'PPI': ppi,
    'Cpu brand': label_encoders["Cpu brand"].transform([cpu])[0],
    'HDD': hdd,
    'SSD': ssd,
    'Gpu brand': label_encoders["Gpu brand"].transform([gpu])[0],
    'os': label_encoders["os"].transform([os])[0],
}

input_df = pd.DataFrame([input_dict])

# Keep only numerical/scaled features for scaler
scaled_input = input_df[['Ram', 'Weight', 'PPI', 'HDD', 'SSD']]
scaled_input = scaler.transform(scaled_input)

# Prediction button
if st.button("Predict Price 💰"):
    try:
        # Predict with both models
        log_pred_lgbm = lgbm_model.predict(scaled_input)[0]
        log_pred_xgb = xgb_model.predict(scaled_input)[0]

        # Average prediction (log -> normal)
        final_log_price = (log_pred_lgbm + log_pred_xgb) / 2
        predicted_price = np.exp(final_log_price)

        st.success(f"Estimated Laptop Price: ₹ {predicted_price:.2f}")
    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction: {str(e)}")
