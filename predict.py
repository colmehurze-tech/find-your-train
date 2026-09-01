import os
import math
import joblib
import pandas as pd

from catboost import CatBoostRegressor, CatBoostClassifier

DELAY_MODEL_PATH = "models/delay_model.cbm"
CAUSE_MODEL_PATH = "models/cause_model.cbm"
METADATA_PATH = "models/model_metadata.pkl"

# LOAD MODELS
print("=" * 60)
print("LOADING TRAINED MODELS")
print("=" * 60)

delay_model = CatBoostRegressor()
delay_model.load_model(DELAY_MODEL_PATH)

cause_model = CatBoostClassifier()
cause_model.load_model(CAUSE_MODEL_PATH)

metadata = joblib.load(METADATA_PATH)

FEATURES = metadata["features"]
CATEGORICAL_FEATURES = metadata["categorical_features"]

print("Delay model loaded.")
print("Cause model loaded.")
print("\nModels are ready!\n")

# HELPER FUNCTIONS
def time_to_minutes(time_string):
    """
    Convert HH:MM into minutes after midnight.
    """

    hours, minutes = map(
        int,
        time_string.strip().split(":")
    )

    return hours * 60 + minutes


def minutes_to_time(total_minutes):
    """
    Convert minutes after midnight back into HH:MM.

    Handles crossing midnight.
    """

    total_minutes = int(round(total_minutes))

    total_minutes %= 1440

    hours = total_minutes // 60
    minutes = total_minutes % 60

    return f"{hours:02d}:{minutes:02d}"


def get_time_of_day(minutes):

    hour = int(minutes // 60)

    if 5 <= hour < 12:
        return "MORNING"

    elif 12 <= hour < 17:
        return "AFTERNOON"

    elif 17 <= hour < 21:
        return "EVENING"

    else:
        return "NIGHT"

# GET USER INPUT
print("=" * 60)
print("TRAIN ETA PREDICTION")
print("=" * 60)

print("\nEnter the journey information.\n")


departure_station = input(
    "Departure station: "
).strip()


arrival_station = input(
    "Arrival station: "
).strip()


scheduled_arrival = input(
    "Scheduled arrival time (HH:MM): "
).strip()


scheduled_departure = input(
    "Scheduled departure time (HH:MM): "
).strip()


distance = float(
    input(
        "Distance between stations (km): "
    )
)


average_speed = float(
    input(
        "Average speed (km/h): "
    )
)


weather = input(
    "Weather condition: "
).strip()


temperature = float(
    input(
        "Temperature (C): "
    )
)


rainfall = float(
    input(
        "Rainfall (mm): "
    )
)


visibility = float(
    input(
        "Visibility (km): "
    )
)


traffic = input(
    "Traffic congestion (Low/Medium/High): "
).strip()


platform = int(
    input(
        "Platform number: "
    )
)
date_string = input(
    "Journey date (YYYY-MM-DD): "
).strip()

# DERIVED FEATURES
date = pd.to_datetime(date_string)

scheduled_arrival_minutes = time_to_minutes(
    scheduled_arrival
)

scheduled_departure_minutes = time_to_minutes(
    scheduled_departure
)

day_of_week = date.day_name()

month = date.month

time_of_day = get_time_of_day(
    scheduled_arrival_minutes
)

# BUILD INPUT DATAFRAME
input_data = {
    "Departure_Station":
        departure_station,

    "Arrival_Station":
        arrival_station,

    "Scheduled_Arrival_Minutes":
        scheduled_arrival_minutes,

    "Scheduled_Departure_Minutes":
        scheduled_departure_minutes,

    "Distance Between Stations (km)":
        distance,

    "Average Speed (km/h)":
        average_speed,

    "Traffic Congestion":
        traffic,

    "Platform Number":
        platform,

    "Weather Condition":
        weather,

    "Temperature (C)":
        temperature,

    "Rainfall (mm)":
        rainfall,

    "Visibility (km)":
        visibility,

    "Day_of_Week":
        day_of_week,

    "Month":
        month,

    "Time_of_Day":
        time_of_day,
}
X = pd.DataFrame(
    [input_data],
    columns=FEATURES
)

# PREPARE CATEGORICAL FEATURES
for column in CATEGORICAL_FEATURES:

    X[column] = (
        X[column]
        .fillna("UNKNOWN")
        .astype(str)
    )

# PREDICT DELAY
predicted_delay = float(
    delay_model.predict(X)[0]
)

# Don't allow a negative delay prediction.
predicted_delay = max(
    0,
    predicted_delay
)



# PREDICT ARRIVAL TIME
predicted_arrival_minutes = (
    scheduled_arrival_minutes
    + predicted_delay
)


predicted_arrival_time = minutes_to_time(
    predicted_arrival_minutes
)

# PREDICT DELAY CAUSE
cause_prediction = (
    cause_model.predict(X)[0][0]
)

# CAUSE CONFIDENCE
probabilities = cause_model.predict_proba(X)[0]

classes = cause_model.classes_

best_probability = max(probabilities)

cause_confidence = (
    best_probability * 100
)



# DISPLAY RESULTS
print("\n")
print("=" * 60)
print("AI TRAIN ETA PREDICTION")
print("=" * 60)

print(
    f"\nFrom              : {departure_station}"
)

print(
    f"To                : {arrival_station}"
)

print(
    f"Scheduled Arrival : {scheduled_arrival}"
)

print(
    f"Predicted Delay   : {predicted_delay:.1f} minutes"
)

print(
    f"Predicted ETA     : {predicted_arrival_time}"
)

print(
    f"Likely Delay Cause: {cause_prediction}"
)

print(
    f"Cause Confidence  : {cause_confidence:.1f}%"
)


print("\n")
print("=" * 60)
print("INPUT CONDITIONS")
print("=" * 60)

print(
    f"Weather           : {weather}"
)

print(
    f"Temperature       : {temperature} °C"
)

print(
    f"Rainfall          : {rainfall} mm"
)

print(
    f"Visibility        : {visibility} km"
)

print(
    f"Traffic           : {traffic}"
)

print(
    f"Distance          : {distance} km"
)

print(
    f"Average Speed     : {average_speed} km/h"
)

print(
    f"Platform          : {platform}"
)


print("\n")
print("=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)