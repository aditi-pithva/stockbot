import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from bayes_opt import BayesianOptimization
import pickle

# Load labeled stock data
df = pd.read_csv("stock_features.csv")  # columns: open, high, low, close, volume, label

# Encode labels
label_map = {"BUY": 0, "HOLD": 1, "SELL": 2}
df["label"] = df["label"].map(label_map)

features = ["open", "high", "low", "close", "volume"]
X = df[features].values
y = df["label"].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for inference
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# Define the model
class StockClassifier(nn.Module):
    def __init__(self, input_dim=5, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.net(x)

# Bayesian Optimization target function
def train_model(lr, dropout):
    model = StockClassifier(dropout=float(dropout))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(5):  # Light training loop for tuning
        model.train()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_logits = model(X_val)
    val_loss = loss_fn(val_logits, y_val).item()
    return -val_loss  # Negative for maximization

# Define search space
pbounds = {
    "lr": (1e-5, 1e-3),
    "dropout": (0.1, 0.5)
}

# Run Bayesian optimization
optimizer = BayesianOptimization(f=train_model, pbounds=pbounds, verbose=2)
optimizer.maximize(init_points=2, n_iter=5)

# Final model training (use best parameters)
best_params = optimizer.max["params"]
final_model = StockClassifier(dropout=best_params["dropout"])
optimizer_final = torch.optim.Adam(final_model.parameters(), lr=best_params["lr"])
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    final_model.train()
    logits = final_model(X_train)
    loss = loss_fn(logits, y_train)
    optimizer_final.zero_grad()
    loss.backward()
    optimizer_final.step()

# Save final model
torch.save(final_model.state_dict(), "models/buy_sell_model.pt")
