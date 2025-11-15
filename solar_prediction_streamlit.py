"""
Solar Power Prediction Streamlit Application
Uses a pre-trained Random Forest model to predict AC power output.
"""

import streamlit as st
import pandas as pd
from joblib import load
import os

# Constants
MODEL_FILE_NAME = 'model.joblib'
DATE_FORMAT = '%Y-%m-%d %H:%M'

MIN_AMBIENT_TEMP = -50.0
MAX_AMBIENT_TEMP = 60.0
MIN_MODULE_TEMP = -50.0
MAX_MODULE_TEMP = 100.0
MIN_IRRADIATION = 0.0
MAX_IRRADIATION = 1.5

# Mappings
GUI_PLANT_ID_MAPPING = {'Plant 1 (ID: 4135001)': 0, 'Plant 2 (ID: 4136001)': 1}

SOURCE_KEY_MAPPING = {
    '1BY6WEcLGh8j5v7': 0, '1IF53ai7Xc0U56Y': 1, '3PZuoBAID5Wc2HD': 2, '7JYdWkrLSPkdwr4': 3,
    'McdE0feGgRqW7Ca': 4, 'VHMLBKoKgIrUVDU': 5, 'WRmjgnKYAwPKWDb': 6, 'WZaEZ7ZgXuqsy4W': 7,
    'ZPfzyrzsUq1vF8S': 8, 'YxYShMdWyrA7czw': 9, 'dDQRgiyyACCcBDH': 10, 'iCRJl6heRkivqQc': 11,
    'ih0MWyTcMoPkXeN': 12, 'mxJmm7HC2YPijMd': 13, 'oZZRqmMfGdntoCq': 14, 'q49JqTeGQLHzLyk': 15,
    'rrqTqKZQfGF4PVE': 16, 'uHbhkAHOPCPlxGn': 17, 'wCzmfRPeisgLqPV': 18, 'z9Y9gH1T5YWrNuG': 19,
    'zBIq5rxdHJRwDNY': 20, 'zVJPv84UY57bAof': 21, '4UPUqMRk7TRMgml': 22, '81aHJ1q11NBPMrL': 23,
    '9kRcWv60rDACzjR': 24, 'Et9kgGMDl729KT4': 25, 'IQ2d7wF4YD8zU1Q': 26, 'LYwnQax7tno7AMq': 27,
    'NX4BCAknbGZzAbt': 28, 'Qf4h9SfABzKipbr': 29, 'QutzIDWKPEPLqvN': 30, 'R9DqYjSHD05jZBK': 31,
    'SMZaZ6WQDfckz4P': 32, 'TRqACzmdXgCzEji': 33, 'V94E5fvQR0Xg2Gf': 34, 'WchCZF8zXeZJrzc': 35,
    'adHrOEFoJhjV6oz': 36, 'dshEAWzCgTdehzi': 37, 'jgaGg8JWf7h0IBz': 38, 'jyJH8pHBYCpMWuJ': 39,
    'qfaP9BGznU8S4nY': 40, 'rGaZhrFxWfASzB7': 41, 'xMbIugepa2P7lBB': 42, 'xoJJ8DcxJEcupym': 43
}

PREDICTION_FEATURES = [
    'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
    'Hour', 'Dayofyear', 'WeekDay', 'PLANT_ID', 'SOURCE_KEY_ENCODED'
]

# --------------------------
# Load Model (FINAL VERSION)
# --------------------------
@st.cache_data
def load_model():
    try:
        if not os.path.exists(MODEL_FILE_NAME):
            st.error(f"Model file '{MODEL_FILE_NAME}' not found.")
            return None

        model = load(MODEL_FILE_NAME)
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# --------------------------
# Validation
# --------------------------
def validate_inputs(date_str, ambient, module, irradiance, plant, inverter):

    try:
        dt = pd.to_datetime(date_str, format=DATE_FORMAT)
    except:
        st.error("Invalid date format. Use YYYY-MM-DD HH:MM")
        return False, None

    if not (MIN_AMBIENT_TEMP <= ambient <= MAX_AMBIENT_TEMP):
        st.error("Ambient temperature out of range.")
        return False, None

    if not (MIN_MODULE_TEMP <= module <= MAX_MODULE_TEMP):
        st.error("Module temperature out of range.")
        return False, None

    if not (MIN_IRRADIATION <= irradiance <= MAX_IRRADIATION):
        st.error("Irradiation out of range.")
        return False, None

    if plant is None or inverter is None:
        st.error("Invalid plant or inverter input.")
        return False, None

    return True, dt


# --------------------------
# Prediction
# --------------------------
def predict(model, dt, ambient, module, irradiance, plant, inverter):

    row = {
        'AMBIENT_TEMPERATURE': ambient,
        'MODULE_TEMPERATURE': module,
        'IRRADIATION': irradiance,
        'Hour': dt.hour,
        'Dayofyear': dt.dayofyear,
        'WeekDay': dt.dayofweek,
        'PLANT_ID': plant,
        'SOURCE_KEY_ENCODED': inverter
    }

    X = pd.DataFrame([row])[PREDICTION_FEATURES]
    pred = float(model.predict(X)[0])

    return max(0.0, pred)


# --------------------------
# MAIN APP
# --------------------------
def main():

    st.set_page_config(page_title="Solar Power Predictor", page_icon="🌞")
    st.title("🌞 Solar AC Power Predictor")

    model = load_model()
    if model is None:
        st.stop()

    st.header("Input Parameters")

    col1, col2 = st.columns(2)

    with col1:
        date_str = st.text_input("Date & Time", "2020-06-16 12:00")
        plant_disp = st.selectbox("Plant", list(GUI_PLANT_ID_MAPPING.keys()))
        plant_code = GUI_PLANT_ID_MAPPING[plant_disp]

        inverter_disp = st.selectbox("Inverter ID", sorted(SOURCE_KEY_MAPPING.keys()))
        inverter_code = SOURCE_KEY_MAPPING[inverter_disp]

    with col2:
        ambient = st.number_input("Ambient Temp (°C)", value=30.0)
        module = st.number_input("Module Temp (°C)", value=45.0)
        irradiance = st.number_input("Irradiation (kW/m²)", value=0.85)

    if st.button("Predict"):

        valid, dt = validate_inputs(date_str, ambient, module, irradiance, plant_code, inverter_code)

        if valid:
            with st.spinner("Predicting..."):
                power = predict(model, dt, ambient, module, irradiance, plant_code, inverter_code)
            st.success(f"🔋 Predicted AC Power: **{power:.2f} kW**")


if __name__ == "__main__":
    main()
