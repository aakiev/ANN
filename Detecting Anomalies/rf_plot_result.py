import pandas as pd
import matplotlib.pyplot as plt

# --- Step 1: Daten laden ---
# Lade den Datensatz
df = pd.read_csv("CSV Files/updated_machine_data_with_predicted_anomalies_rf.csv", index_col="Timestamp", parse_dates=True)

# --- Step 2: Plots vorbereiten ---
fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# --- Plot 1: Ursprüngliche Anomalien ---
axes[0].plot(df.index, df["Temperature"], label="Temperature", alpha=0.7, color="blue")
true_anomalies = df[df["Anomaly"] == 1]
axes[0].scatter(true_anomalies.index, true_anomalies["Temperature"], color="red", label="True Anomalies", s=50)
axes[0].set_title("Temperature with True Anomalies")
axes[0].set_ylabel("Temperature")
axes[0].legend()
axes[0].grid()

# --- Plot 2: Modellvorhersage ---
axes[1].plot(df.index, df["Temperature"], label="Temperature", alpha=0.7, color="blue")
predicted_anomalies = df[df["Predicted_Anomaly"] == 1]  
axes[1].scatter(predicted_anomalies.index, predicted_anomalies["Temperature"], color="orange", label="Predicted Anomalies", s=50)
axes[1].set_title("Temperature with Predicted Anomalies")
axes[1].set_xlabel("Timestamp")
axes[1].set_ylabel("Temperature")
axes[1].legend()
axes[1].grid()

# --- Plots anzeigen ---
plt.tight_layout()
plt.show()
