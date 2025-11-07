"""
Solar Power Prediction Streamlit Application
Uses a pre-trained Random Forest model to predict AC power output from solar panels.
"""

import streamlit as st
import pandas as pd
import pickle
import os
from typing import Dict, Any, Optional, Tuple

# --- Constants ---
MODEL_FILE_NAME = 'solar_prediction_model.pkl'
DATE_FORMAT = '%Y-%m-%d %H:%M'
MIN_AMBIENT_TEMP = -50.0
MAX_AMBIENT_TEMP = 60.0
MIN_MODULE_TEMP = -50.0
MAX_MODULE_TEMP = 100.0
MIN_IRRADIATION = 0.0
MAX_IRRADIATION = 1.5

# --- Mappings Extracted from the Training Step ---
GUI_PLANT_ID_MAPPING = {'Plant 1 (ID: 4135001)': 0, 'Plant 2 (ID: 4136001)': 1}

# All 44 inverter SOURCE_KEYs and their encoded values.
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


@st.cache_data
def load_model() -> Optional[Dict[str, Any]]:
    """
    Load the trained model from the PKL file with caching.
    
    Returns:
        Dictionary containing the model, or None if loading fails
    """
    try:
        if not os.path.exists(MODEL_FILE_NAME):
            st.error(f"Model file not found: '{MODEL_FILE_NAME}'. Please run the ML training script first.")
            return None
        
        with open(MODEL_FILE_NAME, 'rb') as file:
            model_data = pickle.load(file)
        
        # Validate model structure
        if not isinstance(model_data, dict) or 'model' not in model_data:
            st.error("Invalid model file structure. Expected dictionary with 'model' key.")
            return None
        
        return model_data
    except pickle.UnpicklingError as e:
        st.error(f"Failed to unpickle model file: {e}. The file may be corrupted.")
        return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def validate_inputs(
    date_time_str: str,
    ambient_temp: float,
    module_temp: float,
    irradiation: float,
    plant_id_encoded: Optional[int],
    source_key_encoded: Optional[int]
) -> Tuple[bool, Optional[pd.Timestamp]]:
    """
    Validate all input values.
    
    Returns:
        Tuple of (is_valid, datetime_object) or (False, None) if validation fails
    """
    # Validate date/time format
    try:
        dt_object = pd.to_datetime(date_time_str, format=DATE_FORMAT)
    except ValueError:
        st.error(f"Invalid date/time format. Please use: {DATE_FORMAT.replace('%', '')} (e.g., 2020-05-15 12:30)")
        return False, None
    
    # Validate temperature ranges
    if not (MIN_AMBIENT_TEMP <= ambient_temp <= MAX_AMBIENT_TEMP):
        st.error(f"Ambient temperature must be between {MIN_AMBIENT_TEMP}°C and {MAX_AMBIENT_TEMP}°C.")
        return False, None
    
    if not (MIN_MODULE_TEMP <= module_temp <= MAX_MODULE_TEMP):
        st.error(f"Module temperature must be between {MIN_MODULE_TEMP}°C and {MAX_MODULE_TEMP}°C.")
        return False, None
    
    # Validate irradiation range
    if not (MIN_IRRADIATION <= irradiation <= MAX_IRRADIATION):
        st.error(f"Irradiation must be between {MIN_IRRADIATION} and {MAX_IRRADIATION} kW/m².")
        return False, None
    
    # Validate encoding mappings
    if plant_id_encoded is None:
        st.error("Please select a plant location.")
        return False, None
    
    if source_key_encoded is None:
        st.error("Please select an inverter ID.")
        return False, None
    
    return True, dt_object


def predict_power(
    model: Any,
    dt_object: pd.Timestamp,
    ambient_temp: float,
    module_temp: float,
    irradiation: float,
    plant_id_encoded: int,
    source_key_encoded: int
) -> float:
    """
    Make a prediction using the loaded model.
    
    Returns:
        Predicted AC power in kW
    """
    try:
        input_data = {
            'AMBIENT_TEMPERATURE': ambient_temp,
            'MODULE_TEMPERATURE': module_temp,
            'IRRADIATION': irradiation,
            'Hour': dt_object.hour,
            'Dayofyear': dt_object.dayofyear,
            'WeekDay': dt_object.dayofweek,  # 0=Monday, 6=Sunday
            'PLANT_ID': plant_id_encoded,
            'SOURCE_KEY_ENCODED': source_key_encoded
        }
        
        # Create DataFrame in the exact feature order the model was trained on
        X_predict = pd.DataFrame([input_data])[PREDICTION_FEATURES]
        
        prediction = model.predict(X_predict)[0]
        
        # Clip the prediction to 0 as power cannot be negative
        return max(0.0, float(prediction))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return 0.0


# --- Main Application ---
def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Solar Power Prediction",
        page_icon="🌞",
        layout="centered"
    )
    
    st.title("🌞 Solar AC Power Predictor")
    st.markdown("Predict AC Power output using environmental parameters and system configuration")
    st.divider()
    
    # Load model
    model_data = load_model()
    if model_data is None:
        st.stop()
    
    model = model_data['model']
    
    # Sidebar for additional info
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This app uses a pre-trained Random Forest model to predict "
            "solar panel AC power output based on environmental conditions."
        )
        st.markdown("---")
        st.caption(f"Model loaded from: `{MODEL_FILE_NAME}`")
    
    # Input Section
    st.header("📊 Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        date_time = st.text_input(
            "Date and Time",
            value="2020-06-16 12:00",
            help=f"Format: {DATE_FORMAT.replace('%', '')} (e.g., 2020-06-16 12:00)"
        )
        
        plant_display = st.selectbox(
            "Plant Location",
            options=list(GUI_PLANT_ID_MAPPING.keys()),
            help="Select the solar plant location"
        )
        plant_id_encoded = GUI_PLANT_ID_MAPPING.get(plant_display)
        
        inverter_key = st.selectbox(
            "Inverter ID (Source Key)",
            options=sorted(list(SOURCE_KEY_MAPPING.keys())),
            help="Select the inverter/source key"
        )
        source_key_encoded = SOURCE_KEY_MAPPING.get(inverter_key)
    
    with col2:
        ambient_temp = st.number_input(
            "Ambient Temperature (°C)",
            min_value=MIN_AMBIENT_TEMP,
            max_value=MAX_AMBIENT_TEMP,
            value=30.0,
            step=0.1,
            help=f"Range: {MIN_AMBIENT_TEMP} to {MAX_AMBIENT_TEMP}°C"
        )
        
        module_temp = st.number_input(
            "Module Temperature (°C)",
            min_value=MIN_MODULE_TEMP,
            max_value=MAX_MODULE_TEMP,
            value=45.0,
            step=0.1,
            help=f"Range: {MIN_MODULE_TEMP} to {MAX_MODULE_TEMP}°C"
        )
        
        irradiation = st.number_input(
            "Irradiation (kW/m²)",
            min_value=MIN_IRRADIATION,
            max_value=MAX_IRRADIATION,
            value=0.85,
            step=0.01,
            help=f"Range: {MIN_IRRADIATION} to {MAX_IRRADIATION} kW/m²"
        )
    
    st.divider()
    
    # Prediction Section
    if st.button("🔮 Predict AC Power", type="primary", use_container_width=True):
        # Validate inputs
        is_valid, dt_object = validate_inputs(
            date_time,
            ambient_temp,
            module_temp,
            irradiation,
            plant_id_encoded,
            source_key_encoded
        )
        
        if not is_valid or dt_object is None:
            return
        
        # Make prediction
        with st.spinner("Calculating prediction..."):
            prediction = predict_power(
                model,
                dt_object,
                ambient_temp,
                module_temp,
                irradiation,
                plant_id_encoded,
                source_key_encoded
            )
        
        # Display result
        st.success(f"**Predicted AC Power: {prediction:.2f} kW**")
        
        # Additional info
        with st.expander("📋 Prediction Details"):
            st.write(f"**Date/Time:** {dt_object.strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**Plant:** {plant_display}")
            st.write(f"**Inverter:** {inverter_key}")
            st.write(f"**Ambient Temperature:** {ambient_temp:.1f}°C")
            st.write(f"**Module Temperature:** {module_temp:.1f}°C")
            st.write(f"**Irradiation:** {irradiation:.2f} kW/m²")


if __name__ == "__main__":
    main()
