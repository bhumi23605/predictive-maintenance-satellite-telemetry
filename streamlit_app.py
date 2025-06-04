import joblib
import numpy as np
import streamlit as st

# Load model
model = joblib.load("bestmodel.pkl")  # or just "best_model.pkl" if it's in root

st.title("Satellite Predictive Maintenance")

EXPECTED_FEATURES = 6  # Replace with your model's input feature count

st.write(f"Enter {EXPECTED_FEATURES} features:")
inputs = [st.number_input(f"Feature {i+1}") for i in range(EXPECTED_FEATURES)]

if st.button("Predict"):
    features = np.array(inputs).reshape(1, -1)
    
    try:
        prediction = model.predict(features)[0]
        st.success(f"Prediction: {prediction}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
'''      
num_features = st.number_input("Enter number of features", min_value=1)
inputs = [st.number_input(f"Feature {i+1}") for i in range(num_features)]

if st.button("Predict"):
    features = np.array(inputs).reshape(1, -1)
    prediction = model.predict(features)[0]
    st.success(f"Prediction: {prediction}")
'''
