import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# 1️⃣ App title
# -----------------------------
st.title("🏀 NBA Player Next Game Points Predictor")
st.write("Input your player's **last 5 games** to predict the next game points!")

# -----------------------------
# 2️⃣ Load trained model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "random_forest_model.pkl")
model: RandomForestRegressor = joblib.load(model_path)

# -----------------------------
# 3️⃣ Initialize session state for games
# -----------------------------
if "games" not in st.session_state:
    st.session_state.games = pd.DataFrame(columns=["minutes", "points", "FGA", "FGM"])

# -----------------------------
# 4️⃣ Add a new game input (limit to 5 games)
# -----------------------------
st.header("Add a Game")

games_remaining = 5 - len(st.session_state.games)
if games_remaining > 0:
    st.info(f"📊 Games entered: {len(st.session_state.games)}/5 | {games_remaining} more needed")
else:
    st.success("✅ All 5 games entered! You can now predict.")

if len(st.session_state.games) < 5:
    with st.form("add_game_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            minutes = st.number_input("Minutes", value=35.0, min_value=0.0, max_value=48.0)
            points = st.number_input("Points", value=25, min_value=0, max_value=100)
        
        with col2:
            FGA = st.number_input("Field Goals Attempted (FGA)", value=20, min_value=0)
            FGM = st.number_input("Field Goals Made (FGM)", value=10, min_value=0)
        
        submitted = st.form_submit_button("➕ Add Game")
        
        if submitted:
            new_game = {"minutes": minutes, "points": points, "FGA": FGA, "FGM": FGM}
            st.session_state.games = pd.concat([st.session_state.games, pd.DataFrame([new_game])], ignore_index=True)
            st.success(f"Game added! ({len(st.session_state.games)}/5)")
            st.rerun()
else:
    st.warning("🔒 You've entered 5 games. Delete a game below if you want to add a different one.")

# -----------------------------
# 5️⃣ Display games table
# -----------------------------
st.header("Games Entered")
if len(st.session_state.games) > 0:
    display_df = st.session_state.games.copy()
    display_df.insert(0, 'Game #', range(1, len(display_df) + 1))
    st.dataframe(display_df, use_container_width=True)
    
    if st.button("🗑️ Clear All Games"):
        st.session_state.games = pd.DataFrame(columns=["minutes", "points", "FGA", "FGM"])
        st.rerun()
else:
    st.info("No games entered yet. Add your first game above!")

# -----------------------------
# 6️⃣ Predict button
# -----------------------------
st.header("Make Prediction")

if len(st.session_state.games) == 5:
    if st.button("🎯 Predict Next Game Points", type="primary", use_container_width=True):
        df = st.session_state.games.copy()
        
        # Rolling features
        df["avg_pts_last_5"] = df["points"].rolling(window=5).mean().shift(1)
        df["avg_min_last_5"] = df["minutes"].rolling(window=5).mean().shift(1)
        df["trend_pts"] = df["points"].shift(1) - df["avg_pts_last_5"]
        
        df = df.fillna(0)
        latest_game = df.iloc[-1]
        
        # Features for prediction
        features = ["avg_pts_last_5", "avg_min_last_5", "trend_pts", "minutes", "FGA", "FGM"]
        X_new = latest_game[features].values.reshape(1, -1)
        
        predicted_points = model.predict(X_new)[0]
        
        # Display prediction
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                label="Predicted Points for Next Game",
                value=f"{predicted_points:.1f}",
                delta=f"{predicted_points - df['points'].mean():.1f} vs avg"
            )
        
        # Show last 5 games statistics
        st.markdown("---")
        st.subheader("📈 Player Statistics (Last 5 Games)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Points", f"{df['points'].mean():.1f}")
        with col2:
            st.metric("Avg Minutes", f"{df['minutes'].mean():.1f}")
        with col3:
            st.metric("Avg FGA", f"{df['FGA'].mean():.1f}")
        with col4:
            st.metric("Avg FGM", f"{df['FGM'].mean():.1f}")
        
        # Show features used for model
        with st.expander("🔍 View Model Input Features"):
            feature_df = pd.DataFrame({
                'Feature': features,
                'Value': [f"{val:.2f}" for val in X_new[0]]
            })
            st.dataframe(feature_df, use_container_width=True)
else:
    st.warning(f"⚠️ Please enter exactly 5 games to make a prediction. ({len(st.session_state.games)}/5 entered)")
