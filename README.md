Solar Power Prediction using Machine Learning

This project predicts solar AC power output based on environmental and plant-level parameters using a Random Forest Regression model.
It includes both a Tkinter desktop GUI and a Streamlit web interface for interactive predictions.

What is AC Power Output?

AC Power Output (in kilowatts, kW) is the final usable power that comes out of the inverter after it converts DC to AC.

It represents how much electrical energy (in real-world usable form) your solar plant is actually generating at a given time.

⚙️ Example

Suppose your solar plant has panels with:
Ambient Temperature = 30°C
Module Temperature = 45°C
Irradiation = 0.85 kW/m²

Your model might predict:
🧠 Predicted AC Power: 315.42 kW

That means — at that exact time and weather condition, your plant is producing approximately 315.42 kilowatts of usable AC power, which is being supplied to the grid or facility.

1️⃣ AMBIENT_TEMPERATURE (°C)
Meaning: The temperature of the air around the solar plant.
Why it matters:
High ambient temperature reduces the efficiency of solar panels slightly.
Panels perform best at moderate temperatures (around 25°C).
📘 Example: If it’s 35°C outside, panels get hotter and produce a bit less power.

2️⃣ MODULE_TEMPERATURE (°C)

Meaning: The surface temperature of the solar panel itself.
Why it matters:
It has a direct impact on performance — as module temperature rises, voltage drops, reducing power output.
It’s usually higher than the ambient temperature because the panel absorbs sunlight.
📘 Example: If the air is 30°C, the panel might reach 45–50°C.

3️⃣ IRRADIATION (kW/m²) 🌤️
Meaning: The amount of solar energy (sunlight) falling per square meter of panel surface.
Unit: kilowatt per square meter (kW/m²).
Why it matters:
It’s the most important factor — more sunlight means more energy generation.
When the sun is bright, irradiation might be around 1.0 kW/m².
At cloudy times, it drops (e.g., 0.2–0.5 kW/m²).
📘 Example:
At noon on a clear day → IRRADIATION ≈ 1.0 kW/m²
During sunrise/sunset → IRRADIATION ≈ 0.2–0.4 kW/m²
So, you can think of irradiation = intensity of sunlight on the solar panels.

SOURCE_KEY_ENCODED 
Meaning: Encoded ID for the inverter (the device converting DC to AC).
Example:
Inverter ID 1BY6WEcLGh8j5v7 → encoded as 0
Inverter ID QutzIDWKPEPLqvN → encoded as 30

🧠 **Overview**

This project uses a Random Forest Regression model to predict solar energy generation (AC power) based on:
Ambient Temperature
Module Temperature
Irradiation
Time Features (Hour, Day of Year, Weekday)
Plant ID and Inverter ID
The trained model is saved as a .pkl file for reuse in GUI and web applications.

⚙️ **Features**
✅ Predict solar AC power instantly using user inputs
✅ Two user interfaces:
🖥️ Tkinter GUI App
🌐 Streamlit Web App
✅ Clean, validated input handling
✅ Ready-to-train and deploy model script
✅ Easily extendable to new datasets

🧩 **Technologies Used**
Python 3.10+
Pandas
NumPy
Scikit-Learn
Tkinter (Desktop UI)
Streamlit (Web UI)
Pickle

**Note**
The file solar_prediction_model.pkl is not included in the repository due to its binary size.
To generate it, simply run:
  python solar_prediction_model.py

📜 **License**
This project is released under the MIT License — you’re free to use, modify, and share it with proper attribution.

💡 **Author**

Shubham Patel
🎓 B.Tech in Computer Science & Engineering (Data Science)
💻 Passionate about AI, Machine Learning, and Data Exploration
📧 linkedin Url : https://www.linkedin.com/in/siibhu/

