import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

df = pd.read_csv("data/rajdhani_12313_demo_20k.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

features = [
    "Station_Code",
    "Train_Priority",
    "Weather_Condition",
    "Time_of_Day",
    "Day_of_Week",
    "Network_Congestion",
    "Active_Incident",
    "Scheduled_Arrival_Mins",
    "Distance_From_Source_Km",
]

target = "Delay_Minutes"

X = df[features]
y = df[target]

categorical_features = [
    "Station_Code",
    "Train_Priority",
    "Weather_Condition",
    "Time_of_Day",
    "Day_of_Week",
    "Network_Congestion",
    "Active_Incident",
]

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

model.save_model("catboost_eta_model.cbm")
print("\nModel saved as:")
print("catboost_eta_model.cbm")