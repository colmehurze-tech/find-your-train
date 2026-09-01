import os
import math
import joblib
import pandas as pd

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    classification_report,
)

#CONFIG
CSV_PATH = "data/new_delhi_rajdhani_express_synthetic_dataset.csv"

MODEL_DIR = "models"

DELAY_MODEL_PATH = os.path.join(
    MODEL_DIR, "delay_model.cbm"
)

CAUSE_MODEL_PATH = os.path.join(
    MODEL_DIR, "cause_model.cbm"
)

# LOAD DATASET
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(CSV_PATH)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print("\nColumns:")
for column in df.columns:
    print(" -", column)

# Convert strings such as "N/A" into proper missing values.
df = df.replace(
    ["N/A", "NA", "None", ""],
    pd.NA
)

# SORT JOURNEYS
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Date"]
).reset_index(drop=True)

# CREATE DEPARTURE STATION
df["Departure_Station"] = (
    df.groupby("Date")["Station Name"]
      .shift(1)
)

df["Arrival_Station"] = df["Station Name"]

def time_to_minutes(value):
    """
    Convert HH:MM into minutes after midnight.
    """

    if pd.isna(value):
        return float("nan")

    try:
        hours, minutes = str(value).split(":")[:2]

        return int(hours) * 60 + int(minutes)

    except Exception:
        return float("nan")

df["Scheduled_Arrival_Minutes"] = (
    df["Scheduled Arrival Time"].apply(time_to_minutes)
)

df["Actual_Arrival_Minutes"] = (
    df["Actual Arrival Time"].apply(time_to_minutes)
)

df["Scheduled_Departure_Minutes"] = (
    df["Scheduled Departure Time"].apply(time_to_minutes)
)

df["Actual_Departure_Minutes"] = (
    df["Actual Departure Time"].apply(time_to_minutes)
)

# CALCULATE ARRIVAL DELAY
def calculate_delay(row):
    """
    Calculate actual arrival delay while correctly handling
    midnight crossings.
    """

    scheduled = row["Scheduled_Arrival_Minutes"]
    actual = row["Actual_Arrival_Minutes"]

    if pd.isna(scheduled) or pd.isna(actual):
        return float("nan")

    delay = actual - scheduled

    if delay < -720:
        delay += 1440

    elif delay > 720:
        delay -= 1440

    return max(0, delay)

df["Delay_Minutes"] = df.apply(
    calculate_delay,
    axis=1
)

# CREATE DATE FEATURES
df["Day_of_Week"] = (
    df["Date"].dt.day_name()
)

df["Month"] = (
    df["Date"].dt.month
)

#CREATE TIME-OF-DAY FEATURE
def get_time_of_day(minutes):

    if pd.isna(minutes):
        return "UNKNOWN"

    hour = int(minutes // 60)

    if 5 <= hour < 12:
        return "MORNING"

    elif 12 <= hour < 17:
        return "AFTERNOON"

    elif 17 <= hour < 21:
        return "EVENING"

    else:
        return "NIGHT"

df["Time_of_Day"] = (
    df["Scheduled_Arrival_Minutes"]
    .apply(get_time_of_day)
)

# NUMERIC FEATURES
numeric_columns = [
    "Distance Between Stations (km)",
    "Average Speed (km/h)",
    "Temperature (C)",
    "Rainfall (mm)",
    "Visibility (km)",
    "Platform Number",
    "Scheduled_Arrival_Minutes",
    "Scheduled_Departure_Minutes",
    "Month",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

FEATURES = [

    # Route information
    "Departure_Station",
    "Arrival_Station",

    # Schedule
    "Scheduled_Arrival_Minutes",
    "Scheduled_Departure_Minutes",

    # Operational information
    "Distance Between Stations (km)",
    "Average Speed (km/h)",
    "Traffic Congestion",
    "Platform Number",

    # Weather
    "Weather Condition",
    "Temperature (C)",
    "Rainfall (mm)",
    "Visibility (km)",

    # Time information
    "Day_of_Week",
    "Month",
    "Time_of_Day",
]

CATEGORICAL_FEATURES = [
    "Departure_Station",
    "Arrival_Station",
    "Traffic Congestion",
    "Weather Condition",
    "Day_of_Week",
    "Time_of_Day",
]

#PREPARE DATA
delay_df = df.dropna(
    subset=[
        "Delay_Minutes"
    ]
).copy()


print("\nDelay training rows:", len(delay_df))

X_delay = delay_df[FEATURES].copy()

y_delay = delay_df["Delay_Minutes"]

for column in CATEGORICAL_FEATURES:
    X_delay[column] = (
        X_delay[column]
        .fillna("UNKNOWN")
        .astype(str)
    )

# Numeric missing values are filled with median values.
for column in FEATURES:
    if column not in CATEGORICAL_FEATURES:
        X_delay[column] = pd.to_numeric(
            X_delay[column],
            errors="coerce"
        )
        X_delay[column] = (
            X_delay[column]
            .fillna(X_delay[column].median())
        )

# SPLIT BY DATE
unique_dates = delay_df["Date"].drop_duplicates()

train_dates, test_dates = train_test_split(
    unique_dates,
    test_size=0.20,
    random_state=42
)

train_mask = delay_df["Date"].isin(train_dates)
test_mask = delay_df["Date"].isin(test_dates)

X_train = X_delay.loc[train_mask]
X_test = X_delay.loc[test_mask]

y_train = y_delay.loc[train_mask]
y_test = y_delay.loc[test_mask]

print("\nTrain rows:", len(X_train))
print("Test rows:", len(X_test))

# TRAIN CATBOOST REGRESSOR
print("\n")
print("=" * 60)
print("TRAINING DELAY MODEL")
print("=" * 60)


delay_model = CatBoostRegressor(

    iterations=700,

    depth=8,

    learning_rate=0.05,

    loss_function="RMSE",

    random_seed=42,

    verbose=100
)

delay_model.fit(

    X_train,

    y_train,

    cat_features=CATEGORICAL_FEATURES
)

# 15. TEST DELAY MODEL
delay_predictions = (
    delay_model.predict(X_test)
)
mae = mean_absolute_error(
    y_test,
    delay_predictions
)
rmse = math.sqrt(
    mean_squared_error(
        y_test,
        delay_predictions
    )
)

print("\n")
print("=" * 60)
print("DELAY MODEL RESULTS")
print("=" * 60)

print(
    f"MAE  : {mae:.2f} minutes"
)

print(
    f"RMSE : {rmse:.2f} minutes"
)

#CONVERT DELAY CAUSES INTO BROAD CATEGORIES
print("\n")
print("=" * 60)
print("PREPARING DELAY CAUSE MODEL")
print("=" * 60)


def categorize_delay_cause(cause):
    """
    Categorize the delay cause into one of several broad categories.
    The original dataset contains 1000+ unique text descriptions.
    Many of them are actually the same underlying cause.
    """

    if pd.isna(cause):
        return "NO_SIGNIFICANT_DELAY"

    cause = str(cause).lower().strip()

    weather_keywords = [
        "rain",
        "rainfall",
        "thunderstorm",
        "mist",
        "fog",
        "visibility",
        "weather",
        "storm",
        "flood",
        "waterlogging",
        "heat",
        "cold",
        "wind",
    ]

    if any(word in cause for word in weather_keywords):
        return "WEATHER"

    signal_keywords = [
        "signal failure",
        "signal disruption",
        "signalling",
        "signal",
    ]

    if any(word in cause for word in signal_keywords):
        return "SIGNAL_ISSUE"

    speed_keywords = [
        "speed restriction",
        "speed limit",
        "restricted speed",
    ]

    if any(word in cause for word in speed_keywords):
        return "SPEED_RESTRICTION"

    maintenance_keywords = [
        "maintenance",
        "track work",
        "track maintenance",
        "engineering block",
        "maintenance block",
        "track repair",
        "repair work",
    ]

    if any(word in cause for word in maintenance_keywords):
        return "TRACK_MAINTENANCE"

    congestion_keywords = [
        "connecting train",
        "crossing train",
        "preceding train",
        "late running",
        "traffic congestion",
        "congestion",
        "precedence",
        "crossing",
        "overtaking",
    ]

    if any(word in cause for word in congestion_keywords):
        return "NETWORK_CONGESTION"

    passenger_keywords = [
        "passenger",
        "chain pulling",
        "chain pulling",
        "medical",
        "emergency",
        "unauthorized",
        "boarding",
        "deboarding",
    ]

    if any(word in cause for word in passenger_keywords):
        return "PASSENGER_ISSUE"

    crew_keywords = [
        "crew",
        "loco pilot",
        "driver",
        "staff",
        "crew change",
    ]

    if any(word in cause for word in crew_keywords):
        return "CREW_ISSUE"

    operational_keywords = [
        "operational",
        "platform",
        "yard",
        "shunting",
        "locomotive",
        "engine",
        "railway operation",
        "operational bottleneck",
    ]

    if any(word in cause for word in operational_keywords):
        return "OPERATIONAL_ISSUE"

    crossing_keywords = [
        "level crossing",
        "crossing gate",
        "gate",
    ]

    if any(word in cause for word in crossing_keywords):
        return "LEVEL_CROSSING"

    return "OTHER"

df["Delay_Cause_Category"] = (
    df["Cause of Delay (if any)"]
    .apply(categorize_delay_cause)
)

print("\nOriginal unique causes:")
print(
    df["Cause of Delay (if any)"]
    .nunique(dropna=True)
)

print("\nNew ML categories:")
print(
    df["Delay_Cause_Category"]
    .value_counts()
)

# PREPARE CAUSE MODEL DATA
cause_df = df.dropna(
    subset=[
        "Scheduled_Arrival_Minutes"
    ]
).copy()

X_cause = cause_df[FEATURES].copy()
y_cause = cause_df[
    "Delay_Cause_Category"
].astype(str)

# CAUSE MODEL FEATURES
for column in CATEGORICAL_FEATURES:
    X_cause[column] = (
        X_cause[column]
        .fillna("UNKNOWN")
        .astype(str)
    )

#Numeric columns
for column in FEATURES:
    if column not in CATEGORICAL_FEATURES:
        X_cause[column] = pd.to_numeric(
            X_cause[column],
            errors="coerce"
        )
        X_cause[column] = (
            X_cause[column]
            .fillna(X_cause[column].median())
        )

# SPLIT CAUSE DATA BY DATE
cause_train_dates, cause_test_dates = train_test_split(
    cause_df["Date"].drop_duplicates(),
    test_size=0.20,
    random_state=42
)


cause_train_mask = (
    cause_df["Date"].isin(cause_train_dates)
)

cause_test_mask = (
    cause_df["Date"].isin(cause_test_dates)
)


X_cause_train = X_cause.loc[
    cause_train_mask
]

X_cause_test = X_cause.loc[
    cause_test_mask
]

y_cause_train = y_cause.loc[
    cause_train_mask
]

y_cause_test = y_cause.loc[
    cause_test_mask
]

print("\nCause training rows:")
print(len(X_cause_train))

print("\nCause testing rows:")
print(len(X_cause_test))

#TRAIN CATBOOST CAUSE CLASSIFIER
print("\n")
print("=" * 60)
print("TRAINING DELAY CAUSE MODEL")
print("=" * 60)

cause_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="MultiClass",
    random_seed=42,
    thread_count=2,
    verbose=50
)

cause_model.fit(
    X_cause_train,
    y_cause_train,
    cat_features=CATEGORICAL_FEATURES
)

# EVALUATE CAUSE MODEL
cause_predictions = (
    cause_model.predict(X_cause_test)
    .flatten()
)
cause_accuracy = accuracy_score(
    y_cause_test,
    cause_predictions
)

cause_f1 = f1_score(
    y_cause_test,
    cause_predictions,
    average="weighted"
)

print("\n")
print("=" * 60)
print("CAUSE MODEL RESULTS")
print("=" * 60)
print(
    f"Accuracy : {cause_accuracy:.4f}"
)
print(
    f"F1 Score : {cause_f1:.4f}"
)
print("\nClassification Report:")
print(
    classification_report(
        y_cause_test,
        cause_predictions,
        zero_division=0
    )
)

# SAVE MODELS
os.makedirs(
    MODEL_DIR,
    exist_ok=True
)
delay_model.save_model(
    DELAY_MODEL_PATH
)
cause_model.save_model(
    CAUSE_MODEL_PATH
)

metadata = {
    "features": FEATURES,
    "categorical_features":
        CATEGORICAL_FEATURES,
    "delay_target":
        "Delay_Minutes",
    "cause_target":
        "Delay_Cause_Category",
    "delay_mae":
        float(mae),
    "delay_rmse":
        float(rmse),
    "cause_accuracy":
        float(cause_accuracy),
    "cause_f1":
        float(cause_f1),
    "cause_categories": sorted(
        y_cause.unique().tolist()
    )
}

joblib.dump(
    metadata,
    os.path.join(
        MODEL_DIR,
        "model_metadata.pkl"
    )
)

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nSaved files:")

print(
    f"1. {DELAY_MODEL_PATH}"
)

print(
    f"2. {CAUSE_MODEL_PATH}"
)

print(
    f"3. {MODEL_DIR}/model_metadata.pkl"
)

print("\nCause categories used by the model:")

for category in sorted(
    y_cause.unique()
):
    print(
        f" - {category}"
    )