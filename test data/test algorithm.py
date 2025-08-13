import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Load CSV and preprocess
# ----------------------------
def load_and_preprocess_csv(file_path, scaler, look_back=5):
    df = pd.read_csv(file_path)

    required_cols = {"datetime", "total_displacement"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{file_path} is missing required columns {required_cols}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    start_time = df["datetime"].iloc[0]
    times_ms = (df["datetime"] - start_time).dt.total_seconds() * 1000

    disp = df["total_displacement"].values

    print(f"Actual displacement stats: min={disp.min()}, max={disp.max()}, mean={disp.mean()}")

    disp_scaled = scaler.transform(disp.reshape(-1, 1))

    X = []
    for i in range(len(disp_scaled) - look_back):
        X.append(np.column_stack((times_ms[i:i+look_back], disp_scaled[i:i+look_back])))

    X = np.array(X)
    return X, disp[look_back:]  # aligned actual displacement

# ----------------------------
# Main testing function
# ----------------------------
def main():
    test_file = r"C:\Users\Michele\Desktop\Dataset Graphing\7-9-2017_displacement.csv"
    
    model = load_model("trained_model.h5", compile=False)
    scaler = joblib.load("scaler.pkl")

    try:
        X_test, actual_disp = load_and_preprocess_csv(test_file, scaler)
    except ValueError as e:
        print(e)
        return

    predictions_scaled = model.predict(X_test)

    print(f"Predictions scaled stats before clipping: min={predictions_scaled.min()}, max={predictions_scaled.max()}, mean={predictions_scaled.mean()}")

    # Clip predictions to valid scaler range [0,1]
    predictions_scaled = np.clip(predictions_scaled, 0, 1)

    print(f"Predictions scaled stats after clipping: min={predictions_scaled.min()}, max={predictions_scaled.max()}, mean={predictions_scaled.mean()}")

    predictions = scaler.inverse_transform(predictions_scaled).flatten()

    print(f"Predictions inverse scaled stats: min={predictions.min()}, max={predictions.max()}, mean={predictions.mean()}")

    mae = mean_absolute_error(actual_disp, predictions)
    mse = mean_squared_error(actual_disp, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_disp, predictions)

    print(f"\nModel evaluation on {os.path.basename(test_file)}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

if __name__ == "__main__":
    main()
