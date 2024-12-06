import pandas as pd
import matplotlib.pyplot as plt

# Load the processed data
df = pd.read_csv("CSV Files/processed_machine_data.csv", index_col="Timestamp", parse_dates=True)

# Define a plotting function for cleaner code
def plot_data(df, columns, title, ylabel):
    plt.figure(figsize=(14, 7))
    for col in columns:
        plt.plot(df.index, df[col], label=col)
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()

# Plot original and smoothed data for Temperature
plot_data(
    df=df,
    columns=["Temperature", "Temperature_smoothed"],
    title="Temperature: Original vs. Smoothed",
    ylabel="Temperature"
)

# Plot original and filtered data for Vibration
plot_data(
    df=df,
    columns=["Vibration", "Vibration_filtered"],
    title="Vibration: Original vs. Filtered",
    ylabel="Vibration"
)

# Plot standardized data for RPM
plot_data(
    df=df,
    columns=["RPM_standardized"],
    title="Standardized RPM",
    ylabel="Standardized Value"
)

# Plot normalized data for Pressure
plot_data(
    df=df,
    columns=["Pressure_normalized"],
    title="Normalized Pressure",
    ylabel="Normalized Value"
)

# Plot anomalies (highlight anomalous points)
plt.figure(figsize=(14, 7))
plt.plot(df.index, df["Temperature"], label="Temperature", alpha=0.7)
anomalies = df[df["Anomaly"] == -1]
plt.scatter(anomalies.index, anomalies["Temperature"], color="red", label="Anomalies", s=50)
plt.title("Temperature with Anomalies Highlighted")
plt.xlabel("Timestamp")
plt.ylabel("Temperature")
plt.legend()
plt.grid()
plt.show()
