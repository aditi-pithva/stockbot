import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

LABELS = {0: "BUY", 1: "HOLD", 2: "SELL"}

class StockClassifier(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.net(x)

# Load trained model
model = StockClassifier()
model.load_state_dict(torch.load("models/buy_sell_model.pt", map_location=torch.device("cpu")))
model.eval()

# Load saved scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def get_price_data(ticker):
    """Fetch recent stock price data from yfinance"""
    try:
        df = yf.Ticker(ticker).history(period="5d")
        return df if not df.empty else None
    except Exception as e:
        print("Error fetching price data:", e)
        return None

def predict_stock_trend(prices_df):
    """Run prediction on latest available stock data"""
    if prices_df is None or len(prices_df) == 0:
        return "HOLD"

    try:
        latest = prices_df.iloc[-1][["Open", "High", "Low", "Close", "Volume"]].values.reshape(1, -1)
        latest_scaled = scaler.transform(latest)
        X_tensor = torch.tensor(latest_scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = model(X_tensor)
            prediction = torch.argmax(logits, dim=1).item()
            return LABELS[prediction]
    except Exception as e:
        print("Prediction error:", e)
        return "HOLD"
