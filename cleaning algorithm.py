import pandas as pd
import os

# Input and output paths
input_file = 'Data Preprocessing/Synthetic-Dataset/EHAM_LIMC.csv'
output_file = 'Data Preprocessing/Synthetic-Dataset/processed_data/EHAM_LIMC_clean.csv'

# Make sure output folder exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Read the original CSV
data = pd.read_csv(input_file)

# Remove any rows containing NaN in any column
cleaned = data.dropna()

# Save cleaned CSV with the same column order and format
cleaned.to_csv(output_file, index=False)

print(f"Cleaned file saved to {output_file}")
