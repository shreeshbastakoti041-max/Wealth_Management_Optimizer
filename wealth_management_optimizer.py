# ==============================
# Quant Stock Prediction Project
# ==============================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------
# 1. Download Data
# ------------------------------
ticker = "AAPL"

data = yf.download(
    ticker,
    start="2019-01-01",
    end="2024-01-01",
    auto_adjust=True,
    progress=False
)

# Flatten columns if multi-index (prevents alignment errors)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Keep only Close price
data = data[["Close"]].copy()

# ------------------------------
# 2. Feature Engineering
# ------------------------------

# Moving averages (ensure Series, not DataFrame)
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()

data["Prev_Close"] = data["Close"].shift(1)

# Breakout signal (safe alignment)
data["Breakout"] = (
    (data["Close"] > data["MA50"]) &
    (data["Close"].shift(1) <= data["MA50"].shift(1))
).astype(int)

# Target = Next Day Close
data["Target"] = data["Close"].shift(-1)

# Remove missing values
data.dropna(inplace=True)

# ------------------------------
# 3. Train-Test Split
# ------------------------------

X = data[["Prev_Close", "MA20", "MA50", "Breakout"]]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ------------------------------
# 4. Train Model
# ------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ------------------------------
# 5. Evaluation
# ------------------------------

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.2f}")
print(f"R2: {r2:.4f}")

# ------------------------------
# 6. Strategy Backtest
# ------------------------------

data["Daily_Return"] = data["Close"].pct_change()

# Use previous day's breakout to avoid look-ahead bias
data["Strategy_Return"] = data["Daily_Return"] * data["Breakout"].shift(1)

data["Cumulative_Market"] = (1 + data["Daily_Return"]).cumprod()
data["Cumulative_Strategy"] = (1 + data["Strategy_Return"]).cumprod()

# ------------------------------
# 7. Plot Strategy Performance
# ------------------------------

plt.figure()
plt.plot(data["Cumulative_Market"], label="Buy & Hold")
plt.plot(data["Cumulative_Strategy"], label="Breakout Strategy")
plt.legend()
plt.title("Strategy Performance")
plt.show()

# ------------------------------
# 8. Plot Prediction vs Actual
# ------------------------------

plt.figure()
plt.plot(y_test.values, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Next-Day Close Prediction")
plt.show()
