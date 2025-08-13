import os 
import glob
import random
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import joblib

# ----------------------------
# Load all CSV files
# ----------------------------
def load_all_csv(folder_path):
    all_times = []
    all_disp = []

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    for file in csv_files:
        df = pd.read_csv(file)

        # Check for necessary columns
        required_cols = {"datetime", "accel_x", "accel_y", "accel_z", "disp_x", "disp_y", "disp_z", "total_displacement"}
        if not required_cols.issubset(df.columns):
            print(f"Skipping {file} (missing required columns)")
            continue

        # Convert datetime column to milliseconds since start
        df["datetime"] = pd.to_datetime(df["datetime"])
        start_time = df["datetime"].iloc[0]
        times_ms = (df["datetime"] - start_time).dt.total_seconds() * 1000

        # Use total_displacement column directly for displacement
        disp = df["total_displacement"].values

        all_times.append(times_ms.values)
        all_disp.append(disp)

    return all_times, all_disp


# ----------------------------
# Random interval resampling
# ----------------------------
def preprocess_data(all_times, all_disp, min_interval_ms=1000, max_interval_ms=10000, tolerance=50):
    combined_time = np.concatenate(all_times)
    combined_disp = np.concatenate(all_disp)

    # Sort by time
    sorted_idx = np.argsort(combined_time)
    combined_time = combined_time[sorted_idx]
    combined_disp = combined_disp[sorted_idx]

    resampled_times = []
    resampled_disp = []

    current_time = combined_time[0]
    end_time = combined_time[-1]

    while current_time <= end_time:
        # Find nearest real data point within tolerance
        mask = np.abs(combined_time - current_time) <= tolerance
        if np.any(mask):
            avg_val = combined_disp[mask].mean()
            resampled_times.append(current_time)
            resampled_disp.append(avg_val)

        # Jump ahead randomly
        step = random.randint(min_interval_ms, max_interval_ms)
        current_time += step

    return np.array(resampled_times), np.array(resampled_disp)


# ----------------------------
# Create input sequences
# ----------------------------
def create_dataset(times, disp, look_back=5):
    X, y = [], []
    for i in range(len(disp) - look_back):
        X.append(np.column_stack((times[i:i+look_back], disp[i:i+look_back])))
        y.append(disp[i+look_back])
    return np.array(X), np.array(y)


# ----------------------------
# Train the model
# ----------------------------
def main():
    folder_path = r"C:\Users\Michele\Desktop\Dataset Graphing"
    all_times, all_disp = load_all_csv(folder_path)

    if not all_times:
        print("No valid CSV files found.")
        return

    times, disp = preprocess_data(all_times, all_disp)

    # Normalize displacement values
    scaler = MinMaxScaler()
    disp_scaled = scaler.fit_transform(disp.reshape(-1, 1))

    # Create dataset
    X, y = create_dataset(times, disp_scaled)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    # Build LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)

    # Save model and scaler
    model.save("trained_model.h5")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and scaler saved.")

if __name__ == "__main__":
    main()
