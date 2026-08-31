import pandas as pd
from catboost import CatBoostRegressor


# -----------------------------------
# 1. Load trained model
# -----------------------------------

model = CatBoostRegressor()

model.load_model("catboost_eta_model_new.cbm")


# -----------------------------------
# 2. Get input
# -----------------------------------

station = input("Station Name: ")

distance = float(input("Distance Between Stations (km): "))
speed = float(input("Average Speed (km/h): "))

weather = input("Weather Condition: ")

temperature = float(input("Temperature (C): "))
rainfall = float(input("Rainfall (mm): "))
visibility = float(input("Visibility (km): "))

traffic = input("Traffic Congestion: ")

platform = int(input("Platform Number: "))

scheduled_time = input("Scheduled Arrival Time (HH:MM): ")


# -----------------------------------
# 3. Convert scheduled time
# -----------------------------------

time = pd.to_datetime(
    scheduled_time,
    format="%H:%M"
)

scheduled_minutes = (
    time.hour * 60
    + time.minute
)


# -----------------------------------
# 4. Create input dataframe
# -----------------------------------

input_data = pd.DataFrame([{
    "Station Name": station,
    "Distance Between Stations (km)": distance,
    "Average Speed (km/h)": speed,
    "Weather Condition": weather,
    "Temperature (C)": temperature,
    "Rainfall (mm)": rainfall,
    "Visibility (km)": visibility,
    "Traffic Congestion": traffic,
    "Platform Number": platform,
    "Scheduled_Arrival_Minutes": scheduled_minutes
}])


# -----------------------------------
# 5. Predict delay
# -----------------------------------

predicted_delay = model.predict(input_data)[0]

predicted_delay = max(0, predicted_delay)


# -----------------------------------
# 6. Calculate ETA
# -----------------------------------

eta_minutes = scheduled_minutes + predicted_delay

eta_hours = int(eta_minutes // 60) % 24
eta_mins = int(eta_minutes % 60)

predicted_eta = f"{eta_hours:02d}:{eta_mins:02d}"


# -----------------------------------
# 7. Delay level
# -----------------------------------

if predicted_delay <= 10:
    delay_level = "LOW"
elif predicted_delay <= 30:
    delay_level = "MEDIUM"
elif predicted_delay <= 60:
    delay_level = "HIGH"
else:
    delay_level = "CRITICAL"


# -----------------------------------
# 8. Display result
# -----------------------------------

print("\n==============================")
print("AI ETA PREDICTION")
print("==============================")

print(f"Predicted Delay : {predicted_delay:.2f} minutes")
print(f"Predicted ETA   : {predicted_eta}")
print(f"Delay Level     : {delay_level}")