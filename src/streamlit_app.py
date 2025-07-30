import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

st.title("💻 Laptop Price Predictor (Log Transformed)")

# Get the current script directory
current_dir = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(current_dir, ".."))

# Load models and encoders
with open(os.path.join(base_path, "gradient_boosting_log_model.pkl"), "rb") as f:
    gb_model = pickle.load(f)

with open(os.path.join(base_path, "lightgbm_log_model.pkl"), "rb") as f:
    lgb_model = pickle.load(f)

with open(os.path.join(base_path, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(base_path, "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

# UI Inputs
company = st.selectbox("Company", options=label_encoders["Company"].classes_)
typename = st.selectbox("Type", options=label_encoders["TypeName"].classes_)
ram = st.selectbox("RAM (GB)", options=[4, 8, 12, 16, 24, 32, 64])
weight = st.number_input("Weight (in kg)", value=2.0)
touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
ips = st.selectbox("IPS Display", ["No", "Yes"])
screen_size = st.number_input("Screen Size (in inches)", value=15.6)
resolution = st.selectbox("Screen Resolution", ["1920x1080", "1366x768", "1600x900", "3840x2160", "3200x1800"])
cpu = st.selectbox("CPU Type", options=label_encoders["Cpu_type"].classes_)
hdd = st.selectbox("HDD (in GB)", options=[0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox("SSD (in GB)", options=[0, 128, 256, 512, 1024])
gpu = st.selectbox("GPU Type", options=label_encoders["Gpu_type"].classes_)
os_type = st.selectbox("Operating System", options=label_encoders["OS"].classes_)

# Feature Engineering
x_res, y_res = map(int, resolution.split('x'))
ppi = ((x_res**2 + y_res**2) ** 0.5) / screen_size

# Prepare input
query = pd.DataFrame({
    "Company": [company],
    "TypeName": [typename],
    "Ram": [ram],
    "Weight": [weight],
    "Touchscreen": [1 if touchscreen == "Yes" else 0],
    "Ips": [1 if ips == "Yes" else 0],
    "ppi": [ppi],
    "Cpu_type": [cpu],
    "HDD": [hdd],
    "SSD": [ssd],
    "Gpu_type": [gpu],
    "OS": [os_type]
})

# Label Encoding
for col in ["Company", "TypeName", "Cpu_type", "Gpu_type", "OS"]:
    le = label_encoders[col]
    query[col] = le.transform(query[col])

# Scale Numerical Features
X_scaled = query.copy()
X_scaled[["Ram", "Weight", "ppi", "HDD", "SSD"]] = scaler.transform(
    query[["Ram", "Weight", "ppi", "HDD", "SSD"]])

# Predict
model_choice = st.radio("Choose Model", ["Gradient Boosting", "LightGBM"])
if st.button("Predict Price"):
    if model_choice == "Gradient Boosting":
        log_price = gb_model.predict(X_scaled)[0]
    else:
        log_price = lgb_model.predict(X_scaled)[0]
    
    predicted_price = np.exp(log_price)
    st.success(f"Estimated Laptop Price: ₹{int(predicted_price)}")
