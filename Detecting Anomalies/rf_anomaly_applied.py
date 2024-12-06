import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- Step 1: Daten laden ---
df = pd.read_csv("CSV Files/processed_machine_data.csv", index_col="Timestamp", parse_dates=True)

# --- Step 2: Feature-Engineering ---
# Differenzen berechnen (Vorherige Werte vergleichen)
df['Temperature_diff'] = df['Temperature'].diff().fillna(0)
df['RPM_diff'] = df['RPM'].diff().fillna(0)
df['Pressure_diff'] = df['Pressure'].diff().fillna(0)
df['Vibration_diff'] = df['Vibration'].diff().fillna(0)

# Prozentuale Änderungen
df['Temperature_pct_change'] = df['Temperature'].pct_change().fillna(0)
df['RPM_pct_change'] = df['RPM'].pct_change().fillna(0)
df['Pressure_pct_change'] = df['Pressure'].pct_change().fillna(0)
df['Vibration_pct_change'] = df['Vibration'].pct_change().fillna(0)

# Mittelwerte und Standardabweichungen über Zeitfenster
window_size = 5
df['Temperature_mean'] = df['Temperature'].rolling(window=window_size).mean().fillna(df['Temperature'].mean())
df['Temperature_std'] = df['Temperature'].rolling(window=window_size).std().fillna(df['Temperature'].std())

# --- Step 3: Features und Labels ---
features = df[['Temperature', 'RPM', 'Pressure', 'Vibration', 
               'Temperature_diff', 'RPM_diff', 'Pressure_diff', 'Vibration_diff',
               'Temperature_pct_change', 'RPM_pct_change', 'Pressure_pct_change', 'Vibration_pct_change',
               'Temperature_mean', 'Temperature_std']]
labels = df['Anomaly']

# --- Step 4: Trainingsdaten vorbereiten ---
# Mischverhältnis von normalen und Anomalie-Daten sicherstellen
normal_data = df[df['Anomaly'] == 0].sample(n=len(df[df['Anomaly'] == 1]) * 10, random_state=42)
anomaly_data = df[df['Anomaly'] == 1]
balanced_data = pd.concat([normal_data, anomaly_data])

X = balanced_data[features.columns]
y = balanced_data['Anomaly']

# --- Step 5: Trainings- und Testdaten ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Step 6: Random Forest trainieren ---
clf = RandomForestClassifier(
    n_estimators=300,  # Mehr Bäume für Stabilität
    random_state=42,
    class_weight={0: 1, 1: 20}  # Gewicht auf Anomalien erhöhen
)
clf.fit(X_train, y_train)

# --- Step 7: Vorhersagen auf gesamten Datensatz ---
df['Predicted_Anomaly'] = clf.predict(features)

# --- Step 8: Neue Anomalien markieren ---
new_anomalies = df[(df['Predicted_Anomaly'] == 1) & (df['Anomaly'] == 0)]

print("Neue Anomalien erkannt:")
print(new_anomalies)

# --- Step 9: Modellbewertung ---
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Step 10: Visualisierung ---
plt.figure(figsize=(14, 7))

# Originaldaten mit echten Anomalien
plt.subplot(2, 1, 1)
plt.plot(df.index, df['Temperature'], label="Temperature", alpha=0.7)
plt.scatter(df[df['Anomaly'] == 1].index, df[df['Anomaly'] == 1]['Temperature'], color='red', label="True Anomalies", s=50)
plt.title("Temperature with True Anomalies")
plt.xlabel("Timestamp")
plt.ylabel("Temperature")
plt.legend()
plt.grid()

# Vorhergesagte Anomalien
plt.subplot(2, 1, 2)
plt.plot(df.index, df['Temperature'], label="Temperature", alpha=0.7)
plt.scatter(df[df['Predicted_Anomaly'] == 1].index, df[df['Predicted_Anomaly'] == 1]['Temperature'], color='orange', label="Predicted Anomalies", s=50)
plt.title("Temperature with Predicted Anomalies")
plt.xlabel("Timestamp")
plt.ylabel("Temperature")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Ergebnisse speichern
#df.to_csv("CSV Files/updated_machine_data_with_predicted_anomalies_rf.csv", index=True)
#if not new_anomalies.empty:
#    new_anomalies.to_csv("CSV Files/new_predicted_anomalies_rf.csv", index=True)
