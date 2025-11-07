import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# --- 1. Data Loading and Initial Cleaning ---

# Load the datasets
gen_1_df = pd.read_csv("dataset/Plant_1_Generation_Data.csv")
weather_1_df = pd.read_csv("dataset/Plant_1_Weather_Sensor_Data.csv")
gen_2_df = pd.read_csv("dataset/Plant_2_Generation_Data.csv")
weather_2_df = pd.read_csv("dataset/Plant_2_Weather_Sensor_Data.csv")

# Standardize DATE_TIME column to datetime objects
# Plant 1 Generation has a unique date format ('%d-%m-%Y %H:%M')
gen_1_df['DATE_TIME'] = pd.to_datetime(gen_1_df['DATE_TIME'], format='%d-%m-%Y %H:%M')
# All other datasets use the default format ('%Y-%m-%d %H:%M:%S')
weather_1_df['DATE_TIME'] = pd.to_datetime(weather_1_df['DATE_TIME'])
gen_2_df['DATE_TIME'] = pd.to_datetime(gen_2_df['DATE_TIME'])
weather_2_df['DATE_TIME'] = pd.to_datetime(weather_2_df['DATE_TIME'])

# --- 2. Data Merging and Combination ---

# Drop the sensor SOURCE_KEY from weather data as it's not needed for merging
weather_1_df_clean = weather_1_df.drop(columns=['SOURCE_KEY'])
weather_2_df_clean = weather_2_df.drop(columns=['SOURCE_KEY'])

# Merge Generation and Weather Data for Plant 1 (on DATE_TIME and PLANT_ID)
plant_1_merged = pd.merge(gen_1_df, weather_1_df_clean, on=['DATE_TIME', 'PLANT_ID'], how='left')

# Merge Generation and Weather Data for Plant 2
plant_2_merged = pd.merge(gen_2_df, weather_2_df_clean, on=['DATE_TIME', 'PLANT_ID'], how='left')

# Combine the two plants' data into a single DataFrame
combined_df = pd.concat([plant_1_merged, plant_2_merged], ignore_index=True)

# --- 3. Feature Engineering and Preprocessing ---

# Handle missing weather values (fill with 0)
weather_cols = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
combined_df[weather_cols] = combined_df[weather_cols].fillna(0)

# Feature Engineering: Time-based features
combined_df['Hour'] = combined_df['DATE_TIME'].dt.hour
combined_df['Dayofyear'] = combined_df['DATE_TIME'].dt.dayofyear
combined_df['WeekDay'] = combined_df['DATE_TIME'].dt.dayofweek # Monday=0, Sunday=6

# Encode PLANT_ID and Generation SOURCE_KEY using category codes
combined_df['PLANT_ID'] = combined_df['PLANT_ID'].astype('category').cat.codes
combined_df['SOURCE_KEY_ENCODED'] = combined_df['SOURCE_KEY'].astype('category').cat.codes

# Define features (X) and target (y)
features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
            'Hour', 'Dayofyear', 'WeekDay', 'PLANT_ID', 'SOURCE_KEY_ENCODED']
target = 'AC_POWER'

X = combined_df[features]
y = combined_df[target]

# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Model Training and Evaluation ---

# Initialize and train the Random Forest Regressor
# A max_depth limit and min_samples_split are added to prevent overfitting
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=5)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f} kW")
print(f"R-squared (R2) Score: {r2:.4f}")