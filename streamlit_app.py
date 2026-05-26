import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import kurtosis

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("bestmodel (1).pkl")

# -----------------------------
# FEATURE EXTRACTION FUNCTION
# -----------------------------
def extract_features(signal):

    signal = np.array(signal)

    # Number of peaks
    peaks, _ = find_peaks(signal)
    n_peaks = len(peaks)

    # Second derivative
    diff2 = np.diff(signal, n=2)

    # Variance of second derivative
    diff2_var = np.var(diff2)

    # Smoothed signal
    smooth_signal = pd.Series(signal).rolling(window=10).mean().dropna()

    smooth_peaks, _ = find_peaks(smooth_signal)
    smooth10_n_peaks = len(smooth_peaks)

    # Variance divided by signal length
    var_div_len = np.var(signal) / len(signal)

    # Peaks in second derivative
    diff2_peaks, _ = find_peaks(diff2)
    diff2_peaks = len(diff2_peaks)

    # Gap squared feature
    if len(peaks) > 1:
        gaps = np.diff(peaks)
        gaps_squared = np.mean(gaps ** 2)
    else:
        gaps_squared = 0

    # Kurtosis
    #kurt = kurtosis(signal)

    features = pd.DataFrame([{
        'n_peaks': n_peaks,
        'diff2_var': diff2_var,
        'smooth10_n_peaks': smooth10_n_peaks,
        'var_div_len': var_div_len,
        'diff2_peaks': diff2_peaks,
        'gaps_squared': gaps_squared
    }])

    return features


def create_windows(signal, window_size=2048, step_size=512):

    windows = []

    for i in range(0, len(signal) - window_size, step_size):

        window = signal[i:i+window_size]
        windows.append(window)

    return windows

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Predictive Maintenance Dashboard")

st.write("Upload a CSV file containing sensor signal data.")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(df.head())

    # Assume first column contains signal
    signal = df.iloc[:,0].dropna().values
    if len(signal) < 20:
        st.error("Signal too short.")
        st.stop()

    # Plot signal
    st.subheader("Signal Plot")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(signal)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")

    st.pyplot(fig)

    # Extract features
    #features = extract_features(signal)
    windows = create_windows(signal)

    results = []

    for window in windows:
    
        features = extract_features(window)
    
        prob = model.predict_proba(features)[0][1]
    
        results.append(prob)
    #st.subheader("Extracted Features")
    #st.write(features)

    # Prediction
    #prediction = model.predict(features)[0]

    # Probability
    #probability = model.predict_proba(features)[0][1]

    st.subheader("Anomaly Probability Trend")

    fig, ax = plt.subplots(figsize=(10,4))
    
    ax.plot(results)
    
    ax.set_xlabel("Window Number")
    ax.set_ylabel("Anomaly Probability")
    
    st.pyplot(fig)


    # Health score
    health_score = int((1 - probability) * 100)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"Anomaly Detected ⚠")
    else:
        st.success("Machine Healthy ✅")

    st.write(f"Anomaly Probability: {probability:.2f}")
    st.write(f"Health Score: {health_score}%")

    # Gauge-style progress
    st.progress(health_score / 100)

    # Maintenance suggestion
    st.subheader("Maintenance Suggestion")

    if probability > 0.8:
        st.error("Immediate inspection recommended.")
    elif probability > 0.5:
        st.warning("Monitor machine condition closely.")
    else:
        st.success("Machine operating normally.")
