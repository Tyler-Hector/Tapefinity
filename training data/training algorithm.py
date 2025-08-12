import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import joblib
import glob
import os

# Load CSV files
folder_path = r"C:\Users\Michele\Desktop\Dataset Graphing"
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

df_list = [pd.read_csv(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)

# Select displacement-related features
features = ['disp_x', 'disp_y', 'disp_z', 'total_displacement']
data = df[features]

# Scale features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Save scaler for future predictions
joblib.dump(scaler, "human_walk_scaler.pkl")

# Create sequences for LSTM
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])  # Next timestep prediction
    return np.array(X), np.array(y)

SEQ_LENGTH = 10
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Train-test split (time-series safe: no shuffle)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(SEQ_LENGTH, len(features)), activation='tanh'),
    Dense(len(features))  # Predict same number of features
])

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Save trained model
model.save("human_walk_predictor.keras")

print("Model training complete. Saved as human_walk_predictor.keras")
