import os
import paho.mqtt.client as mqtt
import time
import pyarrow.parquet as pq
import pandas as pd


print(f"Publication")

# ── Config ────────────────────────────────────────────
MQTT_BASE_TOPIC = "reed/machine/M1/workorder/OF10001/group/G1/data"
start_time = pd.Timestamp("2025-09-01 06:15:00", tz="UTC")
end_time = pd.Timestamp("2025-09-01 06:30:00", tz="UTC")
SPEED_FACTOR = 0.5   # 10 = 10x plus rapide que temps réel, 1 = temps réel

file_path = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..", "data", "cnc",
    "OF10001", "OF10001_G_BQC_S8CF2G_TYZBPS.parquet"
))

# ── Connexion MQTT ────────────────────────────────────
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect("localhost", 1883, 60)

# ── Lecture du Parquet ────────────────────────────────
try:
    table = pq.read_table(file_path, filters=[
        ('timestamp', '>=', start_time),
        ('timestamp', '<=', end_time),
    ])
    data = table.to_pandas()
except Exception:
    df = pd.read_parquet(file_path)
    data = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

data = data.sort_values('timestamp')
print(f"Lignes à publier : {len(data)}")

# ── Publication : chaque ligne = toutes ses colonnes publiées,
#    puis on attend l'écart réel vers la ligne suivante / SPEED_FACTOR
for i, row in data.iterrows():
    for column in data.columns:
        if column != "timestamp":
            client.publish(f"{MQTT_BASE_TOPIC}/{column}", str(row[column]))

    print(f"[{i+1}/{len(data)}] timestamp={row['timestamp']}")

    time.sleep(SPEED_FACTOR)

client.disconnect()
print("Fin de la simulation.")
