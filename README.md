# NBA Player Next Game Points Predictor

An end-to-end Machine Learning project that predicts an NBA player's next game scoring performance using historical game statistics and rolling feature engineering.

Built using Python, Scikit-learn, and Streamlit.

## 📌 Project Overview

This project simulates a real-world sports analytics workflow.
The system predicts how many points a player will score in their next game based on their previous 5 games.

The pipeline includes:

Raw data generation

Feature engineering (rolling averages & trends)

Random Forest regression modeling

Model evaluation

Model serialization

Interactive web deployment with Streamlit

# 🧠 Problem Statement

Can we predict a player's next game scoring output using recent performance trends?

Instead of using simple averages, this project applies:

Rolling window statistics

Performance trend analysis

Shot volume metrics (FGA, FGM)

Context features (home/away)

## 🏗️ Project Structure

NBA-player-prediction/
│
├── data/
│   ├── raw/
│   │   └── fake_nba_data.csv
│   └── processed/
│       └── featured_data.csv
│
├── models/
│   └── random_forest_model.pkl
│
├── src/
│   ├── generate_data.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md

## ⚙️ Feature Engineering

The model uses the following engineered features:

Feature	Description
avg_pts_last_5	Rolling average of points over last 5 games
avg_min_last_5	Rolling average of minutes
trend_pts	Difference between last game and rolling average
minutes	Minutes played
home_game	1 = Home, 0 = Away
FGA	Field Goal Attempts
FGM	Field Goal Made

Rolling statistics are shifted to prevent data leakage.

## 🤖 Model

### Model Used:

Random Forest Regressor

### Why Random Forest?

Handles nonlinear relationships

Robust to outliers

Works well with structured tabular data

## 📊 Model Evaluation Metrics

The model is evaluated using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

## 🚀 Streamlit App

### The interactive app allows users to:

Input last 5 games manually

Automatically compute rolling features

Predict next game scoring

View statistical summary

### To run locally:

streamlit run app/app.py

## 📦 Installation

### Clone the repository:

git clone https://github.com/JoemarDeVera/nba-player-prediction.git
cd NBA-player-prediction


### Install dependencies:

pip install -r requirements.txt


Run training:

python src/train_model.py


### Run the app:

streamlit run app/app.py

## 💼 Skills Demonstrated

Data Cleaning & Validation

Time-Series Feature Engineering

Rolling Window Calculations

Trend Modeling

Preventing Data Leakage

Random Forest Regression

Model Evaluation

Model Serialization (Joblib)

Interactive ML Deployment (Streamlit)

Project Structuring & Reproducibility

## 📈 Future Improvements

Add real NBA API integration

Hyperparameter tuning

Feature importance visualization

Player-specific models

Deploy to Streamlit Cloud

## 👤 Author
Joemar Garcia
