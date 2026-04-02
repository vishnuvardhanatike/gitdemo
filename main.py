from fastapi import FastAPI
import joblib
import pandas as pd
import json
from collections import defaultdict, deque

app = FastAPI()

# -----------------------------
# Load Trained AI Model
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# Load Village Grid (OSM Data)
# -----------------------------
with open("village_grid.json", "r") as f:
    data = json.load(f)

poles = data["poles"]
connections = data["connections"]

# -----------------------------
# Build Graph Adjacency
# -----------------------------
adj = defaultdict(list)
for a, b in connections:
    adj[a].append(b)
    adj[b].append(a)


# -----------------------------
# Predict Endpoint
# -----------------------------
@app.get("/predict")
def predict(
    rainfall: float,
    wind_speed: float,
    humidity: float = 70,
    temperature: float = 30
):

    results = {}
    failed_nodes = set()

    # MUST MATCH training feature order EXACTLY
    expected_columns = [
        "age",
        "soil_type",
        "material",
        "elevation_exposure",
        "minor_damage_count",
        "rainfall",
        "wind",
        "humidity",
        "temperature",
        "critical"
    ]

    for pole in poles:

        # Construct model input
        data_dict = {
            "age": pole.get("age", 10),
            "soil_type": pole.get("soil", 1),
            "material": pole.get("material", 1),
            "elevation_exposure": pole.get("elevation_exposure", 0.5),
            "minor_damage_count": pole.get("minor_damage_count", 1),
            "rainfall": rainfall,
            "wind": wind_speed,
            "humidity": humidity,
            "temperature": temperature,
            "critical": pole.get("critical", 0)
        }

        input_df = pd.DataFrame([data_dict])

        # 🔥 Enforce correct feature order
        input_df = input_df[expected_columns]

        # Predict failure probability
        failure_prob = float(model.predict_proba(input_df)[0][1])
        failure = failure_prob > 0.5

        results[pole["id"]] = {
            "id": pole["id"],
            "lat": pole["lat"],
            "lon": pole["lon"],
            "failure_probability": failure_prob,
            "population": pole.get("population", 0),
            "critical": pole.get("critical", 0),
            "failed": failure
        }

        if failure:
            failed_nodes.add(pole["id"])

    # -----------------------------
    # Cascading Failure (BFS)
    # -----------------------------
    queue = deque(failed_nodes)

    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if neighbor not in failed_nodes:
                failed_nodes.add(neighbor)
                queue.append(neighbor)

    # Mark cascaded failures
    for node in failed_nodes:
        results[node]["failed"] = True

    return {
        "poles": list(results.values()),
        "connections": connections
    }