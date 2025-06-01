import streamlit as st
import requests

st.title("Satellite Predictive Maintenance")

num_features = st.number_input("Enter number of features", min_value=1, max_value=100)
inputs = [st.number_input(f"Feature {i+1}") for i in range(num_features)]


if st.button("Predict with XGBoost"):
    response = requests.post("http://localhost:8000/predict/xgb", json={"features": inputs})
    st.write(response.json())

if st.button("Predict with LightGBM"):
    response = requests.post("http://localhost:8000/predict/lgb", json={"features": inputs})
    st.write(response.json())
