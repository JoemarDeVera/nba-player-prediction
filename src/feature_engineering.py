import pandas as pd
import os

# Load raw data
df = pd.read_csv("C:/Users/User/Desktop/NBA player prediction/data/raw/fake_nba_data.csv")

# Convert date column
df["game_date"] = pd.to_datetime(df["game_date"])

# Sort properly
df = df.sort_values(by=["player_name", "game_date"])

# -----------------------------
# Rolling Feature Engineering
# -----------------------------
df["avg_pts_last_5"] = df.groupby("player_name")["points"].rolling(5).mean().shift(1).reset_index(level=0, drop=True)
df["avg_min_last_5"] = df.groupby("player_name")["minutes"].rolling(5).mean().shift(1).reset_index(level=0, drop=True)
df["avg_FGA_last_5"] = df.groupby("player_name")["FGA"].rolling(5).mean().shift(1).reset_index(level=0, drop=True)
df["avg_FGM_last_5"] = df.groupby("player_name")["FGM"].rolling(5).mean().shift(1).reset_index(level=0, drop=True)
df["trend_pts"] = df.groupby("player_name")["points"].shift(1) - df["avg_pts_last_5"]

# Target
df["target_points"] = df["points"]

# Drop NaNs (first few games)
df = df.dropna()

# Save processed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)
processed_path = os.path.join(processed_dir, "featured_data.csv")
df.to_csv(processed_path, index=False)

print("Feature engineering complete!")
print(df.head())
