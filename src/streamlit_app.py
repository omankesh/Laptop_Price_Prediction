import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import os

# Get current and parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Define paths
xgb_model_path = os.path.join(parent_dir, "xgboost_log_model.pkl")
lgbm_model_path = os.path.join(parent_dir, "lightgbm_log_model.pkl")
scaler_path = os.path.join(parent_dir, "scaler.pkl")
label_encoders_path = os.path.join(parent_dir, "label_encoders.pkl")
weights_path = os.path.join(parent_dir, "ensemble_weights.json")

# Load files
try:
    with open(xgb_model_path, "rb") as f:
        xgb_model = pickle.load(f)
    with open(lgbm_model_path, "rb") as f:
        lgbm_model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(label_encoders_path, "rb") as f:
        label_encoders = pickle.load(f)
    with open(weights_path, "r") as f:
        weights = json.load(f)
except FileNotFoundError as e:
    st.error(f"❌ Error loading model/preprocessing files: {e}")
    st.stop()

# Streamlit App UI
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")
st.title("💻 Laptop Price Predictor (Ensemble Model)")
st.markdown("Predict laptop prices using XGBoost and LightGBM models.")

# User Input
company = st.selectbox('Brand (Company)', label_encoders['Company'].classes_)
typename = st.selectbox('Laptop Type', label_encoders['TypeName'].classes_)
ram = st.slider('RAM (GB)', 2, 64, step=2)
weight = st.number_input('Weight (kg)', min_value=0.5, max_value=4.0, step=0.1)
cpu_type = st.selectbox('CPU Type', label_encoders['Cpu_type'].classes_)
touchscreen = st.radio('Touchscreen?', ['No', 'Yes'])
ips = st.radio('IPS Display?', ['No', 'Yes'])
hdd = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (GB)', [0, 128, 256, 512, 1024])
gpu_type = st.selectbox('GPU Type', label_encoders['Gpu_type'].classes_)
os = st.selectbox('Operating System', label_encoders['OS'].classes_)

# Display Details
st.markdown("### 📺 Display Details (PPI Auto Calculated)")
screen_size = st.selectbox('Screen Size (inches)', [13.3, 14.0, 15.6, 16.0, 17.3])

# ✅ FIXED: Resolution selection
resolution_options = {
    'HD (1366x768)': (1366, 768),
    'Full HD (1920x1080)': (1920, 1080),
    '2K (2560x1440)': (2560, 1440),
    '4K (3840x2160)': (3840, 2160)
}
selected_label = st.selectbox('Screen Resolution', list(resolution_options.keys()))
res_width, res_height = resolution_options[selected_label]

# Calculate PPI
ppi_value = round((res_width**2 + res_height**2) ** 0.5 / screen_size, 2)
st.caption(f"🔍 **Calculated PPI**: {ppi_value} based on resolution {res_width}x{res_height} and screen size {screen_size}")

# Prepare Input Data
input_dict = {
    'Company': company,
    'TypeName': typename,
    'Ram': ram,
    'Weight': weight,
    'Cpu_type': cpu_type,
    'Touchscreen': 1 if touchscreen == 'Yes' else 0,
    'IPS': 1 if ips == 'Yes' else 0,
    'ppi': ppi_value,
    'HDD': hdd,
    'SSD': ssd,
    'Gpu_type': gpu_type,
    'OS': os
}

# Create DataFrame and Apply Label Encoding
input_df = pd.DataFrame([input_dict])
for col in label_encoders:
    input_df[col] = label_encoders[col].transform(input_df[col])

# Scale Numeric Features
scaled_input = scaler.transform(input_df)

# Model Predictions (Log values)
log_pred_xgb = xgb_model.predict(scaled_input)[0]
log_pred_lgbm = lgbm_model.predict(scaled_input)[0]

# Ensemble Prediction
final_log_price = (
    log_pred_xgb * weights['XGBoost'] +
    log_pred_lgbm * weights['LightGBM']
)
final_price = np.exp(final_log_price)

# Output
st.subheader("📈 Predicted Laptop Price:")
st.success(f"💰 ₹ {final_price:,.2f}")

st.markdown("---")
st.caption("Built with ❤️ using XGBoost and LightGBM models in an ensemble format.")
