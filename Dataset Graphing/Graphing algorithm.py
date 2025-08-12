import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load CSV
df = pd.read_csv(r"C:\Users\Michele\Desktop\training data\cleaned walking.csv")

# Output directory
output_dir = r"C:\Users\Michele\Desktop\Dataset Graphing"
os.makedirs(output_dir, exist_ok=True)

for date_value, group in df.groupby('date'):
    # Build datetime string from date + time column
    datetime_str = group['date'].astype(str) + ' ' + group.iloc[:, 1].astype(str)

    # Replace last colon before fractional seconds with a dot
    datetime_str = datetime_str.str.replace(r":(?=\d{6,9}$)", ".", regex=True)

    # Parse to datetime
    time_dt = pd.to_datetime(datetime_str, errors='raise')

    # Add datetime to group for sorting
    group = group.copy()
    group['datetime'] = time_dt

    # Sort chronologically
    group = group.sort_values('datetime').reset_index(drop=True)
    time_dt = group['datetime']

    # Acceleration values
    ax = group.iloc[:, 5].values
    ay = group.iloc[:, 6].values
    az = group.iloc[:, 7].values

    # Delta time in seconds
    dt = (time_dt - time_dt.shift()).dt.total_seconds().fillna(0).values

    # Integrations
    vx = np.cumsum(ax * dt)
    vy = np.cumsum(ay * dt)
    vz = np.cumsum(az * dt)

    dx = np.cumsum(vx * dt)
    dy = np.cumsum(vy * dt)
    dz = np.cumsum(vz * dt)

    total_disp = np.sqrt(dx**2 + dy**2 + dz**2)

    # Output CSV
    out_df = pd.DataFrame({
        'datetime': time_dt,
        'accel_x': ax,
        'accel_y': ay,
        'accel_z': az,
        'disp_x': dx,
        'disp_y': dy,
        'disp_z': dz,
        'total_displacement': total_disp
    })

    output_path = os.path.join(output_dir, f"{date_value.replace('/', '-')}_displacement.csv")
    out_df.to_csv(output_path, index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_dt, total_disp, label=f"Total Displacement - {date_value}")

    plt.xlabel("Time")
    plt.ylabel("Displacement (m, approx)")
    plt.title(f"Estimated Linear Displacement on {date_value}")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print(f"Displacement CSV files saved in: {output_dir}")
