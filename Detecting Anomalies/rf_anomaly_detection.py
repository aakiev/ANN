import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- Step 1: Daten laden ---
df = pd.read_csv("CSV Files/processed_machine_data.csv", index_col="Timestamp", parse_dates=True)

# Zielparameter: Anomalie
df['Label'] = df['Anomaly']  # 1 = Anomalie, 0 = Normal

# --- Step 2: Features und Labels ---
features = df[['Temperature', 'RPM', 'Pressure', 'Vibration']]  
labels = df['Label']

# --- Step 3: Trainings- und Testdaten aufteilen ---
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.05, random_state=42, stratify=labels)

# --- Step 4: Random Forest mit GridSearchCV für Hyperparameter-Tuning ---
# Definiere Hyperparameter-Raster
param_grid = {
    'n_estimators': [50, 100, 200],   # Anzahl Bäume
    'max_depth': [None, 10, 20],      # Maximale Tiefe der Bäume
    'min_samples_split': [2, 5, 10],  # Mindestanzahl an Samples für einen Split
    'min_samples_leaf': [1, 2, 4]     # Mindestanzahl an Samples pro Blatt
}

# Initialisiere Random Forest Classifier
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# GridSearchCV initialisieren
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Bestes Modell aus GridSearchCV extrahieren
best_rf = grid_search.best_estimator_
print(f"Beste Hyperparameter: {grid_search.best_params_}")

# --- Step 5: Modell evaluieren ---
y_pred = best_rf.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Step 6: Visualisierung ---
# Plot tatsächliche Anomalien
plt.figure(figsize=(14, 7))
plt.plot(df.index, df["Temperature"], label="Temperature", alpha=0.7)
actual_anomalies = df[df["Label"] == 1]
plt.scatter(actual_anomalies.index, actual_anomalies["Temperature"], color="red", label="True Anomalies", s=50)

# Plot vorhergesagte Anomalien
predicted_anomalies = df.copy()
predicted_anomalies["Predicted_Label"] = best_rf.predict(features)
predicted_anomalies = predicted_anomalies[predicted_anomalies["Predicted_Label"] == 1]
plt.scatter(predicted_anomalies.index, predicted_anomalies["Temperature"], color="orange", label="Predicted Anomalies", s=50)

plt.title("Temperature with Actual and Predicted Anomalies")
plt.xlabel("Timestamp")
plt.ylabel("Temperature")
plt.legend()
plt.grid()
plt.show()
