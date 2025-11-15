# 🌞 Solar Power Prediction using Machine Learning

This project predicts **solar AC power output** based on environmental and plant-level parameters using a **Random Forest Regression** model.

It includes:
- 🖥️ Tkinter Desktop GUI
- 🌐 Streamlit Web App

Both interfaces use the trained ML model (`model.joblib`) for predictions.

---

# ⚡ What is AC Power Output?

**AC Power Output (kW)** is the usable electrical power produced after the inverter converts DC to AC.  
It represents how much real electricity the solar plant is generating at a specific moment.

### 🔍 Example  
| Parameter | Value |
|----------|--------|
| Ambient Temperature | 30°C |
| Module Temperature | 45°C |
| Irradiation | 0.85 kW/m² |

**Predicted AC Power → 315.42 kW**  
This means the plant is producing ~315 kW of usable electricity at that instant.

---

# 🔬 Input Feature Details

### 1️⃣ AMBIENT_TEMPERATURE (°C)
- Air temperature around the plant.
- High ambient temperature slightly reduces panel efficiency.

### 2️⃣ MODULE_TEMPERATURE (°C)
- Temperature of the panel surface.
- High module temperature reduces output more significantly.

### 3️⃣ IRRADIATION (kW/m²)
- Amount of solar energy received per square meter.
- **Most important factor** influencing power output.

### 4️⃣ SOURCE_KEY_ENCODED
Encoded inverter identification number.  
Example:

| Inverter ID | Encoded |
|-------------|---------|
| 1BY6WEcLGh8j5v7 | 0 |
| QutzIDWKPEPLqvN | 30 |

---

# 🧠 Model Overview

The AC power prediction model uses:

- Ambient Temperature  
- Module Temperature  
- Irradiation  
- Hour of Day  
- Day of Year  
- Weekday  
- Plant ID  
- Inverter ID (encoded)

The Random Forest Regressor provides robust predictions.

✔ The model is saved using **Joblib** as:

```
model.joblib
```

---

# 🚀 Features

- 🔮 Predict AC solar power instantly  
- 🖥️ User-friendly Tkinter Desktop App  
- 🌐 Streamlit web interface  
- ✔ Input validation  
- ✔ Re-trainable model  
- ✔ Supports all 44 inverter IDs  

---

# 🧩 Technologies Used

- Python 3.10+  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  
- Tkinter  
- Streamlit  

---

# ⚠️ About the Model File

The trained model file:

```
model.joblib
```

is **not included** due to GitHub size limits.

To generate it, run:

```bash
python solar_prediction_model.py
```

This will train the ML model and create the `model.joblib` file automatically.

---

# 📜 License

This project is released under the **MIT License**.  
You may freely use, modify, or distribute it with proper attribution.

---

# 👤 Author

**Shubham Patel**  
🎓 B.Tech – Computer Science & Engineering (Data Science)  
💻 Passionate about AI, Machine Learning & Data Science  
🔗 LinkedIn: https://www.linkedin.com/in/siibhu/
