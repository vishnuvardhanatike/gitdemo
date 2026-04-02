import osmnx as ox
import networkx as nx
import numpy as np
import json

# Center of village
center_point = (17.401, 78.506)

# Village radius in meters
radius = 500

print("Downloading road network using radius...")

G = ox.graph_from_point(
    center_point,
    dist=radius,
    network_type="drive"
)

# Convert to undirected
G = nx.Graph(G)

nodes = list(G.nodes(data=True))
edges = list(G.edges())

poles = []
connections = []

for node_id, data in nodes:
    poles.append({
        "id": int(node_id),
        "lat": float(data["y"]),
        "lon": float(data["x"]),
        "age": int(np.random.randint(1, 40)),
        "soil": int(np.random.randint(0, 3)),
        "population": int(np.random.randint(80, 900)),
        "critical": int(np.random.randint(0, 2))
    })

for u, v in edges:
    connections.append((int(u), int(v)))

with open("village_grid.json", "w") as f:
    json.dump({
        "poles": poles,
        "connections": connections
    }, f)

print("Village grid saved successfully.")