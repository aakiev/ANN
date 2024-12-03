
import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt

# Seed für reproduzierbare Ergebnisse
np.random.seed(42)

# Zeitstempel generieren (z. B. stündliche Messungen für 1000 Stunden)
timestamps = pd.date_range("2024-01-01", periods=1000, freq="h")

# Simulierte Temperaturdaten
temperature = (
    70  # Grundtemperatur
    + 5 * np.sin(np.linspace(0, 20, len(timestamps)))  # Saisonale Schwankung
    + np.random.normal(0, 2, len(timestamps))  # Zufälliges Rauschen
)

# Außentemperatur simulieren
outdoor_temperature = (
    10  # Durchschnittliche Außentemperatur
    + 10 * np.sin(np.linspace(0, 10, len(timestamps)))  # Saisonale Schwankung
    + np.random.normal(0, 1, len(timestamps))  # Zufälliges Rauschen
)

# Simulierte Drehzahl (RPM)
rpm = (
    1500  # Grunddrehzahl
    + np.random.normal(0, 50, len(timestamps))  # Zufälliges Rauschen
)

# Vibration hängt von der Drehzahl ab
base_vibration = 5  # Grundvibration
rpm_factor = 0.002  # Stärke der Abhängigkeit von der Drehzahl
vibration = (
    base_vibration 
    + rpm_factor * rpm  # Abhängigkeit von der Drehzahl
    + np.random.normal(0, 0.5, len(timestamps))  # Zufälliges Rauschen
)

# Druck hängt von der Temperatur und Außentemperatur ab
base_pressure = 100  # Basisdruck
temp_factor = 0.5  # Einfluss der Innentemperatur
outdoor_factor = 0.2  # Einfluss der Außentemperatur
pressure = (
    base_pressure 
    + temp_factor * temperature  # Temperaturabhängigkeit
    + outdoor_factor * outdoor_temperature  # Außentemperaturabhängigkeit
    + np.random.normal(0, 5, len(timestamps))  # Zufälliges Rauschen
)

# Langsamer Verschleiß (linearer Anstieg der Vibration)
vibration += np.linspace(0, 2, len(timestamps))  # Langsamer linearer Anstieg

# Schichtwechsel alle 8 Stunden: Drehzahl ändert sich leicht
shift_change_indices = np.arange(0, len(timestamps), step=8)
rpm[shift_change_indices] += np.random.uniform(-100, 100, size=len(shift_change_indices))


# Simuliere Maschinenausfälle
failure_indices = np.random.choice(len(timestamps), size=5, replace=False)

# Bei einem Ausfall wird die Drehzahl, Vibration und Druck auf 0 gesetzt
rpm[failure_indices] = 0
vibration[failure_indices] = 0
pressure[failure_indices] = 0

# Temperaturabkühlung bei Ausfall: Innentemperatur nähert sich der Außentemperatur
cooling_rate = 0.2  # Geschwindigkeit, mit der die Temperatur sinkt
for idx in failure_indices:
    # Setze die Temperatur auf einen Wert, der sich der Außentemperatur nähert
    temperature[idx] = outdoor_temperature[idx] + cooling_rate * (temperature[idx] - outdoor_temperature[idx])

# Zufällige Anomalien in den Daten
anomaly_indices = np.random.choice(len(timestamps), size=int(len(timestamps) * 0.01), replace=False)
temperature[anomaly_indices] += np.random.uniform(10, 20, size=len(anomaly_indices))  # Temperaturspitzen
rpm[anomaly_indices] += np.random.uniform(-300, 300, size=len(anomaly_indices))  # Drehzahlschwankungen
vibration[anomaly_indices] += np.random.uniform(3, 5, size=len(anomaly_indices))  # Vibrationsspitzen
pressure[anomaly_indices] += np.random.uniform(-20, 20, size=len(anomaly_indices))  # Druckanomalien

# Wochenendmuster (reduzierte Aktivität)
weekdays = pd.to_datetime(timestamps).weekday  # Wochentage (0=Montag, ..., 6=Sonntag)
weekend_indices = np.where((weekdays == 5) | (weekdays == 6))  # Samstag und Sonntag
rpm[weekend_indices] -= 200  # Drehzahl sinkt am Wochenende

# Daten in ein DataFrame packen
df = pd.DataFrame({
    "Timestamp": timestamps,
    "Temperature": temperature,
    "OutdoorTemperature": outdoor_temperature,
    "RPM": rpm,
    "Pressure": pressure,
    "Vibration": vibration,
})
df.set_index("Timestamp", inplace=True)

# Daten visualisieren
plt.figure(figsize=(14, 10))
plt.plot(df.index, df["Temperature"], label="Temperature (Innere)", color="blue")
plt.plot(df.index, df["OutdoorTemperature"], label="Temperature (Außen)", color="cyan")
plt.plot(df.index, df["RPM"], label="RPM (Drehzahl)", color="orange")
plt.plot(df.index, df["Vibration"], label="Vibration", color="green")
plt.plot(df.index, df["Pressure"], label="Pressure (Druck)", color="purple")
plt.scatter(df.index[failure_indices], df["Temperature"].iloc[failure_indices], color='red', label="Machine Failures", s=30)
plt.title("Simulierte Produktionsdaten")
plt.xlabel("Zeit")
plt.ylabel("Wert")
plt.legend()
plt.show()

# Daten speichern
df.to_csv("simulated_machine_data_realistic.csv")

# Visualisierung aufteilen
plt.figure(figsize=(14, 15))

# Temperatur (Innere und Äußere)
plt.subplot(3, 1, 1)
plt.plot(df.index, df["Temperature"], label="Temperature (Innere)", color="blue")
plt.plot(df.index, df["OutdoorTemperature"], label="Temperature (Außen)", color="cyan")
plt.plot(df.index, df["Vibration"], label="Vibration", color="green")
plt.scatter(df.index[failure_indices], df["Temperature"].iloc[failure_indices], color='red', label="Machine Failures", s=30)
plt.title("Temperaturdaten")
plt.xlabel("Zeit")
plt.ylabel("Temperatur")
plt.legend()

# Drehzahl (RPM) und Vibration
plt.subplot(3, 1, 2)
plt.plot(df.index, df["RPM"], label="RPM (Drehzahl)", color="orange")
plt.scatter(df.index[failure_indices], df["RPM"].iloc[failure_indices], color='red', label="Machine Failures", s=30)
plt.title("Drehzahl")
plt.xlabel("Zeit")
plt.ylabel("Wert")
plt.legend()

# Druck (Pressure)
plt.subplot(3, 1, 3)
plt.plot(df.index, df["Pressure"], label="Pressure (Druck)", color="purple")
plt.scatter(df.index[failure_indices], df["Pressure"].iloc[failure_indices], color='red', label="Machine Failures", s=30)
plt.title("Druckdaten")
plt.xlabel("Zeit")
plt.ylabel("Druck")
plt.legend()

# Layout optimieren
plt.tight_layout()
plt.show()
