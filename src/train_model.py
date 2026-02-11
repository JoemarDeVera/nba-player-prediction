import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import joblib

# -----------------------------
# 1 Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load processed data
processed_path = os.path.join(BASE_DIR, "data", "processed", "featured_data.csv")
df = pd.read_csv(processed_path)

# Folder to save models
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "random_forest_model.pkl")

print("Processed data loaded:")
print(df.head())

# -----------------------------
# 2 Features & Target
# -----------------------------
# Updated to include FGA and FGM
features = [
    "avg_pts_last_5",
    "avg_min_last_5",
    "trend_pts",
    "minutes",
    "home_game",
    "FGA",
    "FGM",
]

target = "target_points"

# Ensure all features & target are numeric
df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df[target] = pd.to_numeric(df[target], errors='coerce')

# Drop rows that have NaN after conversion
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# -----------------------------
# 3 Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4 Train Random Forest
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,  # you can increase trees for better performance
    max_depth=7,       # optional, controls complexity
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# 5 Predict & Evaluate
# -----------------------------
y_pred = model.predict(X_test)

# Ensure numeric & remove any NaNs
y_test = pd.to_numeric(y_test, errors='coerce')
y_pred = pd.to_numeric(y_pred, errors='coerce')
mask = y_test.notna() & pd.notna(y_pred)
y_test = y_test[mask].astype(float)
y_pred = y_pred[mask].astype(float)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# -----------------------------
# 6 Save Model
# -----------------------------
joblib.dump(model, model_path)
print(f"\nModel saved successfully to: {model_path}")
