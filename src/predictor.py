# src/predictor.py

import torch
import torch.nn as nn
import numpy as np
import pickle
import os

# === Label mapping (match your training)
LABELS = {0: "BUY", 1: "SELL", 2: "HOLD"}

# === Your model class (exactly as trained)
class StockClassifier(nn.Module):
    def __init__(self, input_dim=25, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.net(x)

# === Load scaler
def load_scaler():
    path = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

# === Load model
def load_model():
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tenson.pt')
    model = StockClassifier()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# === Main prediction function
def predict(feature_vector):
    """
    Input: list of 25 raw (unscaled) feature values
    Output: 'BUY', 'SELL', or 'HOLD'
    """
    assert len(feature_vector) == 25, "❌ Feature vector must have 25 values."

    model = load_model()
    scaler = load_scaler()

    # Scale features
    scaled = scaler.transform([feature_vector])
    x_tensor = torch.tensor(scaled, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        logits = model(x_tensor)
        prediction = torch.argmax(logits, dim=1).item()

    return LABELS[prediction]

# === Manual test
if __name__ == "__main__":
    from fetcher import fetch_stock_data
    from utils import generate_feature_vector

    ticker = "AAPL"  # Change this to test other stocks
    df = fetch_stock_data(ticker)
    features = generate_feature_vector(df)

    if features:
        result = predict(features)
        print(f"📈 Prediction for {ticker}: {result}")
    else:
        print("❌ Failed to generate valid features.")