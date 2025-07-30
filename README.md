# 💻 Laptop Price Predictor

This Streamlit app predicts laptop prices using an ensemble model built with XGBoost and LightGBM.

## 🚀 Features

- User-friendly UI to input laptop specifications
- Automatic PPI calculation based on screen resolution and size
- Scaled inputs using pre-trained scaler
- Ensemble prediction using two trained models

## 📁 Project Structure

```
project-root/
├── xgboost_log_model.pkl
├── lightgbm_log_model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── ensemble_weights.json
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── src/
    └── app.py
```

## 🌐 Deployment

Push this repo to GitHub and connect it with [Streamlit Cloud](https://streamlit.io/cloud) for deployment.

---
Built with ❤️ by Ankesh.
