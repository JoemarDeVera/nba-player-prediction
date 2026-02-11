import pandas as pd
import os

# -----------------------------
# Load raw data
# -----------------------------
df = pd.read_csv("C:/Users/User/Desktop/NBA player prediction/data/raw/fake_nba_data.csv")

# Convert game_date to datetime
df["game_date"] = pd.to_datetime(df["game_date"])

# Sort by player and date (VERY IMPORTANT for rolling features)
df = df.sort_values(by=["player_name", "game_date"])

# -----------------------------
# Compute rolling features (last 5 games)
# -----------------------------

# Average points last 5 games
df["avg_pts_last_5"] = (
    df.groupby("player_name")["points"]
    .rolling(window=5)
    .mean()
    .shift(1)
    .reset_index(level=0, drop=True)
)

# Average minutes last 5 games
df["avg_min_last_5"] = (
    df.groupby("player_name")["minutes"]
    .rolling(window=5)
    .mean()
    .shift(1)
    .reset_index(level=0, drop=True)
)

# Trend points: last game points minus avg last 5
df["trend_pts"] = df.groupby("player_name")["points"].shift(1) - df["avg_pts_last_5"]

# -----------------------------
# Target Variable
# -----------------------------
df["target_points"] = df["points"]

# -----------------------------
# Drop rows where rolling features are NaN
# -----------------------------
df = df.dropna()

# -----------------------------
# Save processed data
# -----------------------------
# Dynamically find project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ensure processed folder exists
processed_dir = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)

# Save processed file
processed_path = os.path.join(processed_dir, "featured_data.csv")
df.to_csv(processed_path, index=False)

print(f"Feature engineering complete! Processed file saved to:\n{processed_path}")
print(df.head())
