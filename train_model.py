"""
train_model.py
Trains the model used by the prediction service (stand-in for the Practical 3 model)
and saves the training feature means, which the drift check uses as its reference.
"""
import json
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data and split into train/test sets
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple classifier
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

# Save the model for the service to load
joblib.dump(model, "model.pkl")

# Save per-feature training means (reference values for drift detection)
train_means = np.mean(X_train, axis=0).tolist()
with open("train_stats.json", "w") as f:
    json.dump({"feature_means": train_means}, f, indent=2)
print(f"Training feature means: {[round(m, 3) for m in train_means]}")
