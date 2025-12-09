import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def plot_model(df, symbol, days_ahead):
    df["t"] = np.arange(len(df))
    
    X = df[["t"]].values 
    y = df["Close"].values
    
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(df[["t"]])
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    y_pred = model.predict(X_poly)
    
    last_t = df["t"].iloc[-1]
    future_t = np.arange(last_t + 1, last_t + 1 + days_ahead).reshape(-1, 1)
    
    future_X_poly = poly.transform(future_t)
    future_pred = model.predict(future_X_poly)
    
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq="D")
    
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Close"], label="Actual price")
    plt.plot(df["Date"], y_pred, label="Polynomial Trend (deg 3)", color="orange", linewidth=2)
    plt.plot(future_dates, future_pred, label=f"Future {days_ahead}d Prediction", linestyle="--", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"{symbol} Closing Price Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    