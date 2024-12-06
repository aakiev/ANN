import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- Step 1: Daten laden ---
# Lade den vorbereiteten Datensatz (mit Anomalien und Label)
df = pd.read_csv("CSV Files/processed_machine_data.csv", index_col="Timestamp", parse_dates=True)

# Labels erstellen (1 = Anomalie, 0 = Normal)
df['Label'] = 0  # Default to normal
df.loc[df['Anomaly'] == -1, 'Label'] = 1

# --- Step 2: Features und Labels auswählen ---
features = df[['Temperature', 'RPM', 'Vibration']] 
labels = df['Label']

# --- Step 3: Trainings- und Testdaten aufteilen ---
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# --- Step 4: Random Forest Classifier trainieren ---
clf = RandomForestClassifier(
    n_estimators=100,   # Anzahl der Bäume
    max_depth=None,     # Maximale Tiefe der Bäume (None = unbegrenzt)
    random_state=42,    # Zufallsfaktor für Reproduzierbarkeit
    class_weight='balanced'  # Wichtig bei unbalancierten Daten (Anomalien sind selten)
)

clf.fit(X_train, y_train)

# --- Step 5: Vorhersagen treffen ---
y_pred = clf.predict(X_test)

# --- Step 6: Modell bewerten ---
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Step 7: Feature Importance visualisieren ---
# Feature-Wichtigkeit aus dem Modell extrahieren
importances = clf.feature_importances_
feature_names = features.columns

# Visualisierung der Feature-Wichtigkeit
plt.figure(figsize=(8, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Random Forest")
plt.show()


