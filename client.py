"""
client.py  —  sends traffic to the running service to generate logs.
  Phase 1: normal requests (similar to training data)  -> no drift
  Phase 2: bad requests                                -> ERROR lines
  Phase 3: shifted requests (bigger flowers)           -> drift WARNING
Run app.py first, then:  python client.py
"""
import json
import urllib.request

import numpy as np

URL = "http://127.0.0.1:5000"
rng = np.random.default_rng(0)


def post(payload):
    req = urllib.request.Request(
        URL + "/predict", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def get(path):
    try:
        with urllib.request.urlopen(URL + path) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


# Phase 1: normal traffic, close to the training distribution
print("Phase 1: normal requests")
normal_mean = [5.8, 3.0, 3.8, 1.2]
for _ in range(20):
    feats = (rng.normal(normal_mean, 0.3)).round(2).tolist()
    print(feats, "->", post({"features": feats}))
print("Drift check (normal):", json.dumps(get("/drift"), indent=1))

# Phase 2: invalid requests -> errors
print("\nPhase 2: bad requests")
print(post({"features": [1, 2]}))            # wrong number of features
print(post({"feats": [5.1, 3.5, 1.4, 0.2]}))  # missing key
print(post({"features": ["a", 3, 1, 0.2]}))   # non-numeric
print(post({"features": [-1, 3.0, 1.4, 0.2]}))  # handled, but WARNING

# Phase 3: shifted traffic — petals ~60% larger than in training
print("\nPhase 3: drifted requests")
drift_mean = [7.5, 3.0, 6.5, 2.2]
for _ in range(60):
    feats = (rng.normal(drift_mean, 0.3)).round(2).tolist()
    post({"features": feats})
print("Drift check (shifted):", json.dumps(get("/drift"), indent=1))

print("\nFinal stats:", get("/stats"))
