import joblib
import numpy as np
import streamlit as st

# Load model
model = joblib.load("bestmodel.pkl")  # or just "best_model.pkl" if it's in root

st.title("Satellite Predictive Maintenance")
num_features = st.number_input("Enter number of features", min_value=1)
inputs = [st.number_input(f"Feature {i+1}") for i in range(num_features)]

if st.button("Predict"):
    features = np.array(inputs).reshape(1, -1)
    prediction = model.predict(features)[0]
    st.success(f"Prediction: {prediction}")
