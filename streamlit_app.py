import streamlit as st
import requests

st.title("Satellite Predictive Maintenance")

num_features = st.number_input("Enter number of features", min_value=1, max_value=100)
inputs = [st.number_input(f"Feature {i+1}") for i in range(num_features)]

if st.button("Predict"):
    try:
        response = requests.post("http://localhost:8000/predict", json={"features": inputs})
        if response.status_code == 200:
            st.success(f"Prediction: {response.json()['prediction']}")
        else:
            st.error(f"Server returned error: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
