import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# 1 Paths & Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")

# Load trained model
model: RandomForestRegressor = joblib.load(model_path)
print("Model loaded successfully!")

# -----------------------------
# 2 Input: latest games of a player
# -----------------------------
# Example data: last 5 games
# Replace this with your real stats
data = [
    {"minutes": 35.2, "points": 24, "FGA": 21, "FGM": 11},
    {"minutes": 31.0, "points": 23, "FGA": 19, "FGM": 9},
    {"minutes": 36.2, "points": 25, "FGA": 17, "FGM": 10},
    {"minutes": 35.7, "points": 31, "FGA": 22, "FGM": 7},
    {"minutes": 34.7, "points": 28, "FGA": 14, "FGM": 5},
]

df = pd.DataFrame(data)

# -----------------------------
# 3 Compute Rolling Features
# -----------------------------
df["avg_pts_last_5"] = df["points"].rolling(window=5).mean().shift(1)
df["avg_min_last_5"] = df["minutes"].rolling(window=5).mean().shift(1)
df["trend_pts"] = df["points"].shift(1) - df["avg_pts_last_5"]

# Fill NaN for first few games
df = df.fillna(0)

# Use the last game for prediction
latest_game = df.iloc[-1]

# -----------------------------
# 4 Prepare features for prediction
# -----------------------------
features = ["avg_pts_last_5", "avg_min_last_5", "trend_pts", "minutes", "FGA", "FGM"]
X_new = latest_game[features].values.reshape(1, -1)

# -----------------------------
# 5 Predict next game points
# -----------------------------
predicted_points = model.predict(X_new)[0]
print(f"\nPredicted points for the next game: {predicted_points:.2f}")
