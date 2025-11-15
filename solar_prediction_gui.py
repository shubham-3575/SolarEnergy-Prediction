"""
Solar Power Prediction GUI Application
Uses a pre-trained Random Forest model to predict AC power output from solar panels.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from joblib import load
import os
from typing import Optional, Dict, Any, Tuple

# --- Constants ---
MODEL_FILE_NAME = 'model.joblib'
DATE_FORMAT = '%Y-%m-%d %H:%M'
MIN_AMBIENT_TEMP = -50.0
MAX_AMBIENT_TEMP = 60.0
MIN_MODULE_TEMP = -50.0
MAX_MODULE_TEMP = 100.0
MIN_IRRADIATION = 0.0
MAX_IRRADIATION = 1.5

# --- Mappings Extracted from the Training Step ---
# These are hardcoded into the GUI script for simplicity and standalone execution.
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

# --- Global Variables ---
model_data: Optional[Dict[str, Any]] = None


def load_model(silent: bool = False) -> bool:
    """
    Load the trained model and mappings from the PKL file.
    
    Args:
        silent: If True, suppress success message (useful for startup loading)
    
    Returns:
        True if model loaded successfully, False otherwise
    """
    global model_data
    if model_data is not None:
        return True
    
    try:
        if not os.path.exists(MODEL_FILE_NAME):
            messagebox.showerror(
                "Error",
                f"Model file not found: '{MODEL_FILE_NAME}'.\n"
                "Please run the ML training script first to create this file."
            )
            return False
            
        with open(MODEL_FILE_NAME, 'rb') as file:
            model_data = pickle.load(file)
        
        # Validate model structure
        if not isinstance(model_data, dict) or 'model' not in model_data:
            messagebox.showerror(
                "Error",
                "Invalid model file structure. Expected dictionary with 'model' key."
            )
            model_data = None
            return False
        
        if not silent:
            messagebox.showinfo("Success", "ML Model loaded successfully!")
        return True
    except pickle.UnpicklingError as e:
        messagebox.showerror(
            "Error",
            f"Failed to unpickle model file: {e}\n"
            "The file may be corrupted or in an incompatible format."
        )
        return False
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {e}")
        return False

def validate_inputs() -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Validate and extract all input values from the GUI.
    
    Returns:
        Tuple of (is_valid, input_dict) where input_dict contains validated inputs
        or None if validation fails
    """
    # Read inputs
    datetime_str = entry_datetime.get().strip()
    plant_display_name = combo_plant_id.get()
    inverter_key = combo_source_key.get()
    ambient_temp_str = entry_ambient.get().strip()
    module_temp_str = entry_module.get().strip()
    irradiation_str = entry_irradiation.get().strip()

    # Check for empty fields
    if not all([datetime_str, plant_display_name, inverter_key, 
                ambient_temp_str, module_temp_str, irradiation_str]):
        messagebox.showerror("Input Error", "Please ensure all fields are filled.")
        return False, None

    # Validate numeric inputs
    try:
        ambient_temp = float(ambient_temp_str)
        module_temp = float(module_temp_str)
        irradiation = float(irradiation_str)
    except ValueError:
        messagebox.showerror(
            "Input Error",
            "Please enter valid numbers for temperatures and irradiation."
        )
        return False, None

    # Validate ranges
    if not (MIN_AMBIENT_TEMP <= ambient_temp <= MAX_AMBIENT_TEMP):
        messagebox.showerror(
            "Input Error",
            f"Ambient temperature must be between {MIN_AMBIENT_TEMP}°C and {MAX_AMBIENT_TEMP}°C."
        )
        return False, None

    if not (MIN_MODULE_TEMP <= module_temp <= MAX_MODULE_TEMP):
        messagebox.showerror(
            "Input Error",
            f"Module temperature must be between {MIN_MODULE_TEMP}°C and {MAX_MODULE_TEMP}°C."
        )
        return False, None

    if not (MIN_IRRADIATION <= irradiation <= MAX_IRRADIATION):
        messagebox.showerror(
            "Input Error",
            f"Irradiation must be between {MIN_IRRADIATION} and {MAX_IRRADIATION} kW/m²."
        )
        return False, None

    # Validate date/time format
    try:
        dt_object = pd.to_datetime(datetime_str, format=DATE_FORMAT)
    except ValueError:
        messagebox.showerror(
            "Date/Time Error",
            f"Please use the format: {DATE_FORMAT.replace('%', '')}\n"
            f"Example: 2020-05-15 12:30"
        )
        return False, None

    # Validate encoding mappings
    plant_id_encoded = GUI_PLANT_ID_MAPPING.get(plant_display_name)
    if plant_id_encoded is None:
        messagebox.showerror("Input Error", f"Invalid plant selection: {plant_display_name}")
        return False, None

    source_key_encoded = SOURCE_KEY_MAPPING.get(inverter_key)
    if source_key_encoded is None:
        messagebox.showerror("Input Error", f"Invalid inverter selection: {inverter_key}")
        return False, None

    return True, {
        'datetime': dt_object,
        'ambient_temp': ambient_temp,
        'module_temp': module_temp,
        'irradiation': irradiation,
        'plant_id_encoded': plant_id_encoded,
        'source_key_encoded': source_key_encoded
    }


def predict_power() -> None:
    """Reads GUI inputs, processes them for the model, and makes a prediction."""
    if not load_model():
        return

    # Validate inputs
    is_valid, inputs = validate_inputs()
    if not is_valid or inputs is None:
        return

    # Feature Engineering
    try:
        input_data = {
            'AMBIENT_TEMPERATURE': inputs['ambient_temp'],
            'MODULE_TEMPERATURE': inputs['module_temp'],
            'IRRADIATION': inputs['irradiation'],
            'Hour': inputs['datetime'].hour,
            'Dayofyear': inputs['datetime'].dayofyear,
            'WeekDay': inputs['datetime'].dayofweek,  # 0=Monday, 6=Sunday
            'PLANT_ID': inputs['plant_id_encoded'],
            'SOURCE_KEY_ENCODED': inputs['source_key_encoded']
        }

        # Create DataFrame in the exact feature order the model was trained on
        X_predict = pd.DataFrame([input_data])[PREDICTION_FEATURES]

    except Exception as e:
        messagebox.showerror("Processing Error", f"Failed to process input data: {e}")
        return

    # Make Prediction
    try:
        prediction = model_data['model'].predict(X_predict)[0]
        
        # Clip the prediction to 0 as power cannot be negative
        final_prediction = max(0.0, float(prediction))
        
        result_label.config(
            text=f"Predicted AC Power: {final_prediction:.2f} kW",
            foreground="blue"
        )
    except AttributeError:
        messagebox.showerror(
            "Prediction Error",
            "Model object does not have a 'predict' method. Invalid model type."
        )
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Model prediction failed: {e}")


# --- GUI Setup ---

def create_input_row(parent: ttk.Frame, label_text: str, row: int, 
                     is_dropdown: bool = False, options: Optional[list] = None) -> tk.Widget:
    """
    Helper function to create labelled input rows in the GUI.
    
    Args:
        parent: Parent frame widget
        label_text: Label text for the input
        row: Grid row number
        is_dropdown: If True, create a Combobox; otherwise create an Entry
        options: List of options for dropdown (if is_dropdown is True)
    
    Returns:
        The created widget (Entry or Combobox)
    """
    label = ttk.Label(parent, text=f"{label_text}:")
    label.grid(row=row, column=0, padx=10, pady=5, sticky='w')
    
    if is_dropdown:
        combo = ttk.Combobox(parent, values=options, state="readonly", width=40)
        combo.grid(row=row, column=1, padx=10, pady=5, sticky='ew')
        if options:
            combo.set(options[0])  # Set first option as default
        return combo
    else:
        entry = ttk.Entry(parent, width=40)
        entry.grid(row=row, column=1, padx=10, pady=5, sticky='ew')
        return entry


# Initialize main window
root = tk.Tk()
root.title("Solar Power Prediction GUI (Random Forest)")
root.geometry("600x650")
root.resizable(False, False)

# Styling
style = ttk.Style()
style.configure('TFrame', background='#f0f0f0')
style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
style.configure('TButton', font=('Arial', 10, 'bold'))
root.configure(background='#f0f0f0')

main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill='both', expand=True)
main_frame.columnconfigure(1, weight=1)

# Title
title_label = ttk.Label(
    main_frame,
    text="🌞 Solar AC Power Predictor",
    font=('Arial', 16, 'bold')
)
title_label.grid(row=0, column=0, columnspan=2, pady=10)
ttk.Separator(main_frame, orient='horizontal').grid(
    row=1, column=0, columnspan=2, sticky='ew', pady=5
)

# --- Input Fields ---
row_counter = 2

# Date/Time
entry_datetime = create_input_row(
    main_frame,
    "Date and Time (YYYY-MM-DD HH:MM)",
    row_counter
)
entry_datetime.insert(0, "2020-06-16 12:00")
row_counter += 1

# Plant ID (Dropdown)
plant_options = list(GUI_PLANT_ID_MAPPING.keys())
combo_plant_id = create_input_row(
    main_frame,
    "Plant Location",
    row_counter,
    is_dropdown=True,
    options=plant_options
)
row_counter += 1

# Inverter (Source Key) (Dropdown)
inverter_options = sorted(list(SOURCE_KEY_MAPPING.keys()))
combo_source_key = create_input_row(
    main_frame,
    "Inverter ID (Source Key)",
    row_counter,
    is_dropdown=True,
    options=inverter_options
)
row_counter += 1

# Ambient Temperature
entry_ambient = create_input_row(
    main_frame,
    "Ambient Temperature (°C)",
    row_counter
)
entry_ambient.insert(0, "30.0")
row_counter += 1

# Module Temperature
entry_module = create_input_row(
    main_frame,
    "Module Temperature (°C)",
    row_counter
)
entry_module.insert(0, "45.0")
row_counter += 1

# Irradiation
entry_irradiation = create_input_row(
    main_frame,
    "Irradiation (kW/m²)",
    row_counter
)
entry_irradiation.insert(0, "0.85")
row_counter += 1

# Add padding between inputs and button
ttk.Label(main_frame, text="").grid(row=row_counter, column=0, columnspan=2, pady=5)
row_counter += 1

# --- Prediction Button ---
predict_button = ttk.Button(
    main_frame,
    text="Predict AC Power",
    command=predict_power,
    style='TButton'
)
predict_button.grid(row=row_counter, column=0, columnspan=2, pady=10)
row_counter += 1

# --- Result Label ---
result_label = ttk.Label(
    main_frame,
    text="Predicted AC Power: N/A",
    font=('Arial', 14, 'bold')
)
result_label.grid(row=row_counter, column=0, columnspan=2, pady=10)
row_counter += 1

ttk.Separator(main_frame, orient='horizontal').grid(
    row=row_counter, column=0, columnspan=2, sticky='ew', pady=5
)
row_counter += 1

# Status/Hint Label
hint_label = ttk.Label(
    main_frame,
    text="Model loaded from 'solar_prediction_model.pkl'",
    font=('Arial', 8, 'italic')
)
hint_label.grid(row=row_counter, column=0, columnspan=2, pady=5)

# Load model on startup (silently to avoid popup on startup)
root.after(100, lambda: load_model(silent=True))

# Run the GUI
if __name__ == "__main__":

    root.mainloop()
