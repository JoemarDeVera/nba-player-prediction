import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

players = {
    "Player_A": 28,
    "Player_B": 24,
    "Player_C": 15,
    "Player_D": 18,
    "Player_E": 12
}

teams = ["LAL", "GSW", "BOS", "MIL", "PHX", "DEN", "MIA"]

data = []
start_date = datetime(2024, 10, 1)

for player, base_ppg in players.items():
    current_date = start_date
    for game in range(82):
        minutes = np.random.normal(34, 3)
        opponent = np.random.choice(teams)
        
        # FGA and FGM logic
        FGA = np.random.randint(10, 30)
        FGM = np.random.randint(int(FGA*0.3), FGA+1)  # at least 30% made
        
        points = FGM * 2 + np.random.randint(0, 3)  # roughly points from FGM + some randomness
        rebounds = np.random.normal(6, 2)
        assists = np.random.normal(5, 2)
        fg_pct = FGM / FGA
        
        data.append([
            player,
            current_date,
            opponent,
            round(minutes,1),
            int(points),
            int(rebounds),
            int(assists),
            round(fg_pct,2),
            FGA,
            FGM
        ])
        
        current_date += timedelta(days=int(np.random.choice([1,2,3])))

columns = [
    "player_name",
    "game_date",
    "opponent",
    "minutes",
    "points",
    "rebounds",
    "assists",
    "fg_pct",
    "FGA",
    "FGM"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("C:/Users/User/Desktop/NBA player prediction/data/raw/fake_nba_data.csv", index=False)
print(df.head())
