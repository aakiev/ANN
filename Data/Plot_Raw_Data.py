import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("CSV Files/simulated_machine_data_realistic.csv")

# Daten visualisieren
plt.figure(figsize=(14, 10))
plt.plot(df.index, df["Temperature"], label="Temperature (Innere)", color="blue")
plt.plot(df.index, df["OutdoorTemperature"], label="Temperature (Außen)", color="cyan")
plt.plot(df.index, df["RPM"], label="RPM (Drehzahl)", color="orange")
plt.plot(df.index, df["Vibration"], label="Vibration", color="green")
plt.plot(df.index, df["Pressure"], label="Pressure (Druck)", color="purple")
plt.title("Simulierte Produktionsdaten")
plt.xlabel("Zeit")
plt.ylabel("Wert")
plt.legend()
plt.show()

# Visualisierung aufteilen
plt.figure(figsize=(14, 15))

# Temperatur (Innere und Äußere)
plt.subplot(3, 1, 1)
plt.plot(df.index, df["Temperature"], label="Temperature (Innere)", color="blue")
plt.plot(df.index, df["OutdoorTemperature"], label="Temperature (Außen)", color="cyan")
plt.plot(df.index, df["Vibration"], label="Vibration", color="green")
plt.title("Temperaturdaten")
plt.xlabel("Zeit")
plt.ylabel("Temperatur")
plt.legend()

# Drehzahl (RPM) und Vibration
plt.subplot(3, 1, 2)
plt.plot(df.index, df["RPM"], label="RPM (Drehzahl)", color="orange")
plt.title("Drehzahl")
plt.xlabel("Zeit")
plt.ylabel("Wert")
plt.legend()

# Druck (Pressure)
plt.subplot(3, 1, 3)
plt.plot(df.index, df["Pressure"], label="Pressure (Druck)", color="purple")
plt.title("Druckdaten")
plt.xlabel("Zeit")
plt.ylabel("Druck")
plt.legend()

# Layout optimieren
plt.tight_layout()
plt.show()