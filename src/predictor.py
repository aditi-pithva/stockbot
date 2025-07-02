# src/predictor.py

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import warnings
import json

# Suppress device warnings
warnings.filterwarnings("ignore", message=".*Device set to use cpu.*")

LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}

def load_feature_info():
    try:
        path = os.path.join(os.path.dirname(__file__), "..", "models", "feature_info.json")
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Feature info not found, using fallback")
        return {
            'selected_features': None,
            'input_dim': 19,
            'model_params': {
                'hidden_dim1': 128,
                'hidden_dim2': 64,
                'hidden_dim3': 32,
                'dropout_rate': 0.3
            }
        }

class EnhancedStockPredictor(nn.Module):
    def __init__(self, input_dim=24, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, num_classes=3, dropout_rate=0.3):
        super(EnhancedStockPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.bn3 = nn.BatchNorm1d(hidden_dim3)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(hidden_dim3, num_classes)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = self.input_norm(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        
        return x

def predict(feature_vector, feature_names=None):
    """
    Enhanced prediction function using the new model architecture
    """
    try:
        feature_info = load_feature_info()
        
        expected_features = feature_info['input_dim']
        if len(feature_vector) != expected_features:
            print(f"Feature mismatch: got {len(feature_vector)}, expected {expected_features}")
            if len(feature_vector) < expected_features:
                feature_vector = list(feature_vector) + [0.0] * (expected_features - len(feature_vector))
            else:
                feature_vector = feature_vector[:expected_features]
            print(f"Adjusted to {len(feature_vector)} features")
        
        scaler_path = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        model_params = feature_info['model_params']
        model = EnhancedStockPredictor(
            input_dim=feature_info['input_dim'],
            hidden_dim1=model_params.get('hidden_dim1', 128),
            hidden_dim2=model_params.get('hidden_dim2', 64), 
            hidden_dim3=model_params.get('hidden_dim3', 32),
            dropout_rate=model_params.get('dropout_rate', 0.3)
        )
        
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "tensor.pt")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        scaled_features = scaler.transform(feature_array)
        x_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        
        with torch.no_grad():
            logits = model(x_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item() * 100
        
        result = {
            'prediction': LABELS[prediction],
            'confidence': confidence,
            'probabilities': {
                'SELL': probabilities[0][0].item() * 100,
                'HOLD': probabilities[0][1].item() * 100,
                'BUY': probabilities[0][2].item() * 100
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None

# Manual test
if __name__ == "__main__":
    from fetcher import fetch_stock_data
    from utils import generate_feature_vector

    ticker = "AAPL"
    print(f"Testing enhanced prediction for {ticker}")
    
    # Fetch stock data first
    stock_data = fetch_stock_data(ticker)
    if stock_data is not None:
        # Generate features using enhanced feature engineering
        feature_vector, feature_names = generate_feature_vector(stock_data)
        
        if feature_vector and len(feature_vector) > 0:
            result = predict(feature_vector)
            if result:
                print(f"Prediction for {ticker}: {result['prediction']} ({result['confidence']:.1f}% confidence)")
                print(f"Probabilities: SELL={result['probabilities']['SELL']:.1f}%, HOLD={result['probabilities']['HOLD']:.1f}%, BUY={result['probabilities']['BUY']:.1f}%")
            else:
                print("Prediction failed")
        else:
            print("Failed to generate valid features.")
    else:
        print("Failed to fetch stock data.")