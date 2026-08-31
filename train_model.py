import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

df = pd.read_csv("data/new_delhi_rajdhani_express_synthetic_dataset.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Calculate actual delay correctly
df["Scheduled_Arrival_Time"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Scheduled Arrival Time"].astype(str)
)

df["Actual_Arrival_Time"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Actual Arrival Time"].astype(str)
)

# Handle trains crossing midnight
df.loc[
    df["Actual_Arrival_Time"] < df["Scheduled_Arrival_Time"],
    "Actual_Arrival_Time"
] += pd.Timedelta(days=1)

df["Delay_Minutes"] = (
    df["Actual_Arrival_Time"] -
    df["Scheduled_Arrival_Time"]
).dt.total_seconds() / 60

print("\nDelay statistics:")
print(df["Delay_Minutes"].describe())

print(
    "Missing delay values:",
    df["Delay_Minutes"].isna().sum()
)

# Remove rows where target cannot be calculated
df = df.dropna(subset=["Delay_Minutes"])

target = "Delay_Minutes"

scheduled = pd.to_datetime(
    df["Scheduled_Arrival_Time"]
)

df["Scheduled_Arrival_Minutes"] = (
    scheduled.dt.hour * 60
    + scheduled.dt.minute
)

df["Scheduled_Arrival_Minutes"] = (
    scheduled.dt.hour * 60
    + scheduled.dt.minute
)

features = [
    "Station Name",
    "Distance Between Stations (km)",
    "Average Speed (km/h)",
    "Weather Condition",
    "Temperature (C)",
    "Rainfall (mm)",
    "Visibility (km)",
    "Traffic Congestion",
    "Platform Number",
    "Scheduled_Arrival_Minutes",
]

categorical_features = [
    "Station Name",
    "Weather Condition",
    "Traffic Congestion",
]

# Handle missing categorical values
for col in categorical_features:
    df[col] = df[col].fillna("Unknown").astype(str)

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

model = CatBoostRegressor(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    verbose=100
)

print("\nTraining CatBoost...")
model.fit(
    X_train,
    y_train,
    cat_features=categorical_features
)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = math.sqrt(mean_squared_error(y_test, predictions))

print("\n==============================")
print("MODEL RESULTS")
print("==============================")
print(f"MAE  : {mae:.2f} minutes")
print(f"RMSE : {rmse:.2f} minutes")

model.save_model("catboost_eta_model_new.cbm")
print("\nModel saved as:")
print("catboost_eta_model_new.cbm")