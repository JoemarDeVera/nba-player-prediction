import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Players and base points
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
        # Game features
        minutes = np.random.normal(34, 3)
        home_game = np.random.choice([0,1])
        opponent = np.random.choice(teams)
        
        # Simulate shooting stats
        FGA = np.random.randint(10, 30)  # Field Goals Attempted
        FGM = np.random.randint(int(FGA*0.35), int(FGA*0.65))  # Field Goals Made (35%-65% of attempts)
        
        # Performance logic
        points = FGM * 2 + np.random.choice([0,1,2])  # mostly 2 points per FGM plus randomness
        rebounds = np.random.normal(6, 2)
        assists = np.random.normal(5, 2)
        fg_pct = FGM / FGA if FGA > 0 else 0
        
        data.append([
            player,
            current_date,
            opponent,
            home_game,
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
    "home_game",
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
