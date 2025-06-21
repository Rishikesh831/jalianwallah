import streamlit as st
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils import load_model, load_scaler, scale_features

st.set_page_config(page_title="Jallianwala Bagh Public Interest Predictor", page_icon="🕯️", layout="centered")

# Tribute banner
st.markdown("""
    <div style='background: linear-gradient(90deg, #b22222 0%, #ffffff 50%, #228B22 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: #b22222; text-align: center;'>Jallianwala Bagh Public Interest Predictor</h1>
        <h3 style='color: #333; text-align: center;'>A Machine Learning Tribute to the Martyrs of 1919</h3>
        <p style='color: #444; text-align: center;'>
            This project honors the memory of those who lost their lives in the Jallianwala Bagh massacre.<br>
            Enter the features below to predict public interest around the anniversary.<br>
            <span style='color: #b22222; font-weight: bold;'>Never Forget. Never Again.</span>
        </p>
    </div>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join("models", "jallianwala_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
FEATURES = ['NewsArticles', 'YouTubeUploads', 'WeightedEvents', 'InverseDays', 'DaysSquared']

# User input
st.header("Enter Feature Values")
col1, col2 = st.columns(2)
with col1:
    news = st.number_input("Number of News Articles", min_value=0, max_value=100, value=30)
    yt = st.number_input("YouTube Uploads", min_value=0, max_value=50, value=10)
    weighted_events = st.number_input("Weighted Events", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.3f")
with col2:
    inverse_days = st.number_input("Inverse Days to Anniversary", min_value=0.0, max_value=0.1, value=0.005, step=0.0001, format="%.5f")
    days_squared = st.number_input("Days Squared from Anniversary", min_value=0, max_value=200000, value=10000)

if st.button("Predict Public Interest", help="Click to predict public interest level."):
    w, b = load_model(MODEL_PATH)
    mean, std = load_scaler(SCALER_PATH)
    X = np.array([news, yt, weighted_events, inverse_days, days_squared], dtype=float)
    X_scaled = scale_features(X, mean, std)
    pred = X_scaled @ w + b
    st.markdown(f"<div style='background-color:#b22222;padding:1rem;border-radius:8px;text-align:center;'><span style='color:white;font-size:2rem;'>Predicted Public Interest: <b>{pred:.2f}</b></span></div>", unsafe_allow_html=True)
    st.progress(min(max(pred/100, 0), 1), text="Interest Level")
    if pred > 80:
        st.success("High public interest expected. Consider commemorative events and awareness campaigns.")
    elif pred > 40:
        st.info("Moderate public interest. Some engagement likely.")
    else:
        st.warning("Low public interest. More outreach may be needed.")

st.caption("""
---
Created with ❤️ as a tribute to the martyrs of Jallianwala Bagh. | Project by [Your Name]
""") 