import pandas as pd
import numpy as np

# ===== Load CSV =====
df = pd.read_csv(r"C:\Users\Michele\Desktop\Dataset Graphing\6-30-2017_displacement.csv")

# ===== Convert datetime to seconds =====
df['datetime'] = pd.to_datetime(df['datetime'])
time_seconds = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds().values

# ===== Compute velocity from displacement =====
velocity_from_disp = pd.DataFrame({
    'velocity_x': np.gradient(df['disp_x'], time_seconds),
    'velocity_y': np.gradient(df['disp_y'], time_seconds),
    'velocity_z': np.gradient(df['disp_z'], time_seconds),
})

# ===== Compute acceleration from velocity =====
accel_from_vel = pd.DataFrame({
    'accel_x_from_vel': np.gradient(velocity_from_disp['velocity_x'], time_seconds),
    'accel_y_from_vel': np.gradient(velocity_from_disp['velocity_y'], time_seconds),
    'accel_z_from_vel': np.gradient(velocity_from_disp['velocity_z'], time_seconds),
})

# ===== Combine data =====
df_extended = pd.concat([df, velocity_from_disp, accel_from_vel], axis=1)

# ===== Choose features to analyze =====
features = [
    'accel_x', 'accel_y', 'accel_z',
    'velocity_x', 'velocity_y', 'velocity_z',
    'disp_x', 'disp_y', 'disp_z',
    'accel_x_from_vel', 'accel_y_from_vel', 'accel_z_from_vel',
    'total_displacement'
]

# ===== Basic statistics =====
stats = df_extended[features].describe()

# ===== Custom statistics =====
custom_stats = pd.DataFrame({
    'missing_values': df_extended[features].isnull().sum(),
    'unique_values': df_extended[features].nunique(),
    'variance': df_extended[features].var(),
    'range': df_extended[features].max() - df_extended[features].min()
})

# ===== Combine into one table =====
stats_combined = pd.concat([stats, custom_stats.T])

# ===== Print or save =====
print(stats_combined)
