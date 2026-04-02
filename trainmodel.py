import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

np.random.seed(42)

# -----------------------------
# SETTINGS
# -----------------------------

n_poles = 300
n_days = 365 * 5   # reduce to 5 years (faster, still strong)

# Pole attributes
poles = pd.DataFrame({
    "age": np.random.randint(1, 40, n_poles),
    "soil_type": np.random.randint(0, 3, n_poles),
    "material": np.random.randint(0, 2, n_poles),
    "elevation_exposure": np.random.uniform(0, 1, n_poles),
    "minor_damage_count": np.random.poisson(1.5, n_poles),
    "critical": np.random.randint(0, 2, n_poles)
})

# Expand poles across days (vectorized)
poles_expanded = poles.loc[poles.index.repeat(n_days)].reset_index(drop=True)

# Generate daily weather
rainfall = np.random.gamma(2, 10, len(poles_expanded))
wind = np.random.normal(30, 15, len(poles_expanded))
humidity = np.random.uniform(40, 95, len(poles_expanded))
temperature = np.random.normal(30, 5, len(poles_expanded))

# Logistic probability
logit = (
    0.04 * poles_expanded["age"] +
    0.6 * poles_expanded["soil_type"] +
    0.3 * poles_expanded["elevation_exposure"] +
    0.5 * poles_expanded["minor_damage_count"] +
    0.03 * rainfall +
    0.04 * wind +
    0.01 * humidity -
    0.3 * poles_expanded["material"] +
    0.7 * poles_expanded["critical"] -
    6.5
)

prob = 1 / (1 + np.exp(-logit))

failed = np.random.rand(len(prob)) < prob

# Final dataset
df = poles_expanded.copy()
df["rainfall"] = rainfall
df["wind"] = wind
df["humidity"] = humidity
df["temperature"] = temperature
df["failed"] = failed.astype(int)

# -----------------------------
# TRAIN MODEL
# -----------------------------

X = df.drop("failed", axis=1)
y = df["failed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=120,
    class_weight="balanced",
    n_jobs=-1   # use all CPU cores
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

joblib.dump(model, "model.pkl")

print("Model trained and saved.")