import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy.signal import butter, filtfilt

# --- Step 0: Load the dataset ---

# Load the dataset and set the timestamp as the index
df = pd.read_csv("CSV Files/simulated_machine_data_realistic.csv", index_col="Timestamp", parse_dates=True)

# Display the initial structure of the dataset
print("Initial Data:")
print(df.head())

# --- Step 1: Remove rows where RPM or Vibration is zero ---

# Initial row count
initial_row_count = len(df)

# Filter out rows where RPM or Vibration is 0
df = df[(df["RPM"] != 0) & (df["Vibration"] != 0)]

# Rows removed
filtered_row_count = len(df)
print(f"Rows removed where RPM or Vibration = 0: {initial_row_count - filtered_row_count}")

# --- Step 2: Handle Missing Values ---

# Check for missing values
print("\nMissing Values Per Column:")
print(df.isnull().sum())

# Fill missing values using forward-fill and backward-fill
df = df.ffill()  # Forward-fill missing values
df = df.bfill()  # Backward-fill missing values

# Check again for missing values
print("\nMissing Values After Filling:")
print(df.isnull().sum())

# --- Step 3: Smooth the Data ---

# Apply a rolling window to smooth the data
window_size = 5
df_smoothed = df.rolling(window=window_size).mean()

# Add smoothed columns to the DataFrame with '_smoothed' suffix
for col in df.columns:
    df[f"{col}_smoothed"] = df_smoothed[col]

# --- Step 4: Normalize and Standardize the Data ---

# Standardize the data (mean=0, std=1)
scaler_standard = StandardScaler()
df_standardized = pd.DataFrame(
    scaler_standard.fit_transform(df),
    columns=[f"{col}_standardized" for col in df.columns],
    index=df.index
)

# Normalize the data (scale between 0 and 1)
scaler_minmax = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler_minmax.fit_transform(df),
    columns=[f"{col}_normalized" for col in df.columns],
    index=df.index
)

# Combine normalized and standardized data into the original DataFrame
df = pd.concat([df, df_standardized, df_normalized], axis=1)

# --- Step 5: Remove NaN Values ---

# Check for NaN values after processing
print("\nNaN Check After Preprocessing:")
print(df.isnull().sum())

# Remove any remaining NaN values
df.dropna(inplace=True)

# --- Step 6: Multivariate Anomaly Detection with Isolation Forest ---

# Select relevant features for anomaly detection
features = df[['Temperature', 'RPM', 'Vibration']]  # Add more columns if relevant

# Apply Isolation Forest
iso = IsolationForest(contamination=0.008, random_state=42)  # Best result with this contamination
df['Anomaly'] = iso.fit_predict(features)

# Replace Isolation Forest output with binary values: 1 for anomalies, 0 for normal data
df['Anomaly'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Display detected anomalies
print("\nAnomalies Detected (Multivariate):")
print(df[df['Anomaly'] == 1])

# --- Step 7: Filter High-Frequency Noise ---

# Define a Butterworth low-pass filter function
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Apply the filter to numeric columns
cutoff_frequency = 0.05  # Adjust as needed
sampling_frequency = 1  # Assuming hourly data
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:  # Only apply to numeric columns
        df[f"{col}_filtered"] = butter_lowpass_filter(df[col], cutoff_frequency, sampling_frequency)

# --- Step 8: Save the Processed Data ---

# Save the processed DataFrame to a new CSV file
df.to_csv("CSV Files/processed_machine_data.csv")
print("\nProcessed Data Saved to 'processed_machine_data.csv'")
