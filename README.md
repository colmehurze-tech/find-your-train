# find-your-train
# Indian Railways ETA prediction tool and tracking tool

This project uses CatBoost to predict train arrival delays and classify the likely cause of delay from railway operational, schedule, weather, and traffic data.

## 1. Model Training: `train_model.py`

The training script:
1. Loads the 50K-row railway dataset.
2. Cleans missing and invalid values.
3. Creates `Departure_Station` from the previous station and uses the current station as `Arrival_Station`.
4. Converts scheduled and actual times into minutes.
5. Calculates `Delay_Minutes` from scheduled and actual arrival times.
6. Creates features such as day of week, month, and time of day.
7. Uses station, schedule, weather, traffic, speed, distance, and platform information as model features.
8. Splits data by date to reduce data leakage between training and testing.
9. Trains a `CatBoostRegressor` to predict delay in minutes.
10. Evaluates the delay model using MAE and RMSE.
11. Converts detailed delay descriptions into broad categories such as `WEATHER`, `SIGNAL_ISSUE`, `NETWORK_CONGESTION`, `PASSENGER_ISSUE`, and `TRACK_MAINTENANCE`.
12. Trains a `CatBoostClassifier` to predict the delay category.
13. Evaluates the classifier using accuracy and weighted F1 score.
14. Saves the trained models and metadata.

Generated files:

```text
models/
├── delay_model.cbm
├── cause_model.cbm
└── model_metadata.pkl
```

## 2. Prediction: `predict.py`

The prediction script loads the trained models and accepts journey information from the user.

Inputs:
- Departure station
- Arrival station
- Scheduled arrival/departure time
- Distance
- Average speed
- Weather condition
- Temperature
- Rainfall
- Visibility
- Traffic congestion
- Platform number
- Journey date

The script derives the day of week, month, and time of day, then creates the same feature structure used during training.

The inputs are passed to both models:

```text
Input data
    |
    +--> Delay Model --> Predicted delay
    |
    +--> Cause Model --> Predicted cause
```

The predicted delay is added to the scheduled arrival time to calculate the predicted ETA.

The script outputs:

```text
Predicted Delay
Predicted ETA
Likely Delay Cause
Cause Confidence
```

## 3. Running the Model

Install dependencies:

```bash
pip install pandas scikit-learn catboost joblib
```

Train the models:

```bash
python train_model_new.py
```

Run a prediction:

```bash
python predict.py
```

The trained `.cbm` files are ready to be integrated into the FastAPI backend.
