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

# --- Hardcoded mappings ---
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

# --- Global model object ---
model = None


# -------------------- MODEL LOADER --------------------
def load_model(silent: bool = False) -> bool:
    """Load the trained Random Forest model using joblib."""
    global model

    if model is not None:
        return True

    if not os.path.exists(MODEL_FILE_NAME):
        messagebox.showerror(
            "Model Not Found",
            f"Model file '{MODEL_FILE_NAME}' is missing.\nPlease place model.joblib beside this script."
        )
        return False

    try:
        model = load(MODEL_FILE_NAME)

        if not silent:
            messagebox.showinfo("Success", "Model loaded successfully.")

        return True

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model:\n{e}")
        return False


# -------------------- INPUT VALIDATION --------------------
def validate_inputs() -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Validate user inputs from GUI."""
    datetime_str = entry_datetime.get().strip()
    plant_display_name = combo_plant_id.get()
    inverter_key = combo_source_key.get()
    ambient_temp_str = entry_ambient.get().strip()
    module_temp_str = entry_module.get().strip()
    irradiation_str = entry_irradiation.get().strip()

    if not all([datetime_str, plant_display_name, inverter_key,
                ambient_temp_str, module_temp_str, irradiation_str]):
        messagebox.showerror("Input Error", "All fields are mandatory.")
        return False, None

    try:
        ambient_temp = float(ambient_temp_str)
        module_temp = float(module_temp_str)
        irradiation = float(irradiation_str)
    except ValueError:
        messagebox.showerror("Input Error", "Temperature and irradiation must be numeric.")
        return False, None

    if not (MIN_AMBIENT_TEMP <= ambient_temp <= MAX_AMBIENT_TEMP):
        messagebox.showerror("Input Error", "Invalid ambient temperature range.")
        return False, None

    if not (MIN_MODULE_TEMP <= module_temp <= MAX_MODULE_TEMP):
        messagebox.showerror("Input Error", "Invalid module temperature range.")
        return False, None

    if not (MIN_IRRADIATION <= irradiation <= MAX_IRRADIATION):
        messagebox.showerror("Input Error", "Invalid irradiation range.")
        return False, None

    try:
        dt = pd.to_datetime(datetime_str, format=DATE_FORMAT)
    except:
        messagebox.showerror("Format Error", f"Use format: {DATE_FORMAT}")
        return False, None

    plant_id = GUI_PLANT_ID_MAPPING.get(plant_display_name)
    inverter_encoded = SOURCE_KEY_MAPPING.get(inverter_key)

    return True, {
        "datetime": dt,
        "ambient": ambient_temp,
        "module": module_temp,
        "irradiation": irradiation,
        "plant_id": plant_id,
        "source_key": inverter_encoded
    }


# -------------------- PREDICTION LOGIC --------------------
def predict_power():
    """Prepare input data and produce prediction."""
    if not load_model():
        return

    ok, data = validate_inputs()
    if not ok:
        return

    df = pd.DataFrame([{
        "AMBIENT_TEMPERATURE": data["ambient"],
        "MODULE_TEMPERATURE": data["module"],
        "IRRADIATION": data["irradiation"],
        "Hour": data["datetime"].hour,
        "Dayofyear": data["datetime"].dayofyear,
        "WeekDay": data["datetime"].weekday(),
        "PLANT_ID": data["plant_id"],
        "SOURCE_KEY_ENCODED": data["source_key"]
    }])[PREDICTION_FEATURES]

    try:
        pred = model.predict(df)[0]
        pred = max(0, float(pred))

        result_label.config(
            text=f"Predicted AC Power: {pred:.2f} kW",
            foreground="blue"
        )
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Model failed:\n{e}")


# -------------------- GUI SETUP --------------------
def create_input_row(parent, text, row, is_dropdown=False, options=None):
    label = ttk.Label(parent, text=text + ":")
    label.grid(row=row, column=0, padx=10, pady=5)

    if is_dropdown:
        combo = ttk.Combobox(parent, values=options, width=40, state="readonly")
        combo.grid(row=row, column=1, padx=10, pady=5)
        combo.set(options[0])
        return combo
    else:
        entry = ttk.Entry(parent, width=40)
        entry.grid(row=row, column=1, padx=10, pady=5)
        return entry


root = tk.Tk()
root.title("Solar Power Prediction GUI")
root.geometry("600x650")
root.resizable(False, False)

main = ttk.Frame(root, padding=20)
main.pack(fill="both", expand=True)

row = 0

entry_datetime = create_input_row(main, "Date & Time (YYYY-MM-DD HH:MM)", row)
entry_datetime.insert(0, "2020-06-16 12:00")
row += 1

combo_plant_id = create_input_row(main, "Plant Location", row, True, list(GUI_PLANT_ID_MAPPING.keys()))
row += 1

combo_source_key = create_input_row(main, "Inverter ID", row, True, sorted(SOURCE_KEY_MAPPING.keys()))
row += 1

entry_ambient = create_input_row(main, "Ambient Temperature (°C)", row)
entry_ambient.insert(0, "30")
row += 1

entry_module = create_input_row(main, "Module Temperature (°C)", row)
entry_module.insert(0, "45")
row += 1

entry_irradiation = create_input_row(main, "Irradiation (kW/m²)", row)
entry_irradiation.insert(0, "0.85")
row += 1

ttk.Button(main, text="Predict AC Power", command=predict_power).grid(
    row=row, column=0, columnspan=2, pady=15
)
row += 1

result_label = ttk.Label(main, text="Predicted AC Power: N/A", font=("Arial", 14, "bold"))
result_label.grid(row=row, column=0, columnspan=2, pady=10)
row += 1

ttk.Label(main, text="Model loaded from model.joblib", font=("Arial", 8, "italic")).grid(
    row=row, column=0, columnspan=2
)

root.after(200, lambda: load_model(silent=True))

if __name__ == "__main__":
    root.mainloop()
