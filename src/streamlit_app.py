import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load pre-trained objects
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")  # Dictionary of LabelEncoders for categorical columns

st.title("💻 Laptop Price Predictor")
st.caption("By - Om Ankesh")

# --- Input form ---
with st.form("Laptop Features"):
    company = st.selectbox("Company", ["Dell", "HP", "Apple", "Acer"])
    typename = st.selectbox("Type", ["Ultrabook", "Gaming", "Notebook"])
    ram = st.selectbox("RAM (GB)", [4, 8, 16, 32])
    weight = st.number_input("Weight (kg)", 0.5, 5.0, step=0.1)
    cpu_type = st.selectbox("CPU Brand", ["Intel Core i5", "Intel Core i7", "AMD Ryzen"])
    touchscreen = st.selectbox("Touchscreen", ["Yes", "No"])
    ips = st.selectbox("IPS Display", ["Yes", "No"])
    ppi = st.number_input("PPI (Pixels per inch)", 100.0, 300.0, step=1.0)
    hdd = st.selectbox("HDD (GB)", [0, 500, 1024])
    ssd = st.selectbox("SSD (GB)", [0, 256, 512, 1024])
    gpu_type = st.selectbox("GPU Brand", ["Intel", "Nvidia", "AMD"])
    os = st.selectbox("Operating System", ["Windows", "macOS", "Linux", "No OS"])

    submitted = st.form_submit_button("Predict Price")

if submitted:
    try:
        # Create DataFrame for single input
        input_df = pd.DataFrame([[company, typename, ram, weight, cpu_type, touchscreen,
                                  ips, ppi, hdd, ssd, gpu_type, os]],
                                columns=['Company', 'TypeName', 'Ram', 'Weight', 'Cpu_type',
                                         'Touchscreen', 'IPS', 'ppi', 'HDD', 'SSD', 'Gpu_type', 'OS'])

        # Label Encode using pre-fitted encoders
        for col in encoder:
            le = encoder[col]
            input_df[col] = le.transform(input_df[col])

        # Scale numeric values
        input_scaled = scaler.transform(input_df)

        # Predict log price and convert to real price
        log_price = model.predict(input_scaled)[0]
        predicted_price = np.exp(log_price)

        st.success(f"💰 Predicted Price: ₹{predicted_price:,.0f}")
    
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
