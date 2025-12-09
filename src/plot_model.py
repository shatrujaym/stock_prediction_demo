"""
plot_model.py
-------------
Provides a helper function to fit a simple 3rd-degree polynomial regression
on historical stock closing prices and visualize:

- Actual historical prices
- Fitted polynomial trend over the historical period
- A basic extrapolation for a given number of future days

This is intended as an educational/demo tool and not as a serious
financial forecasting model.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def plot_model(df, symbol, days_ahead):
    """
    Plot the historical closing prices of a stock, a 3rd-degree polynomial
    regression trend line, and a simple extrapolated prediction.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that must contain at least the columns:
        - "Date": datetime-like, representing trading days
        - "Close": float or numeric, closing prices
    symbol : str
        Stock ticker symbol used in the plot title and messages.
    days_ahead : int
        Number of future calendar days to forecast and plot.

    Notes
    -----
    This function uses a 3rd-degree polynomial regression purely for
    demonstration/visualization. It is not suitable for real trading
    decisions or robust forecasting.
    """
    # --- Ensure data is sorted and indexed correctly ---

    # Sort by date to guarantee time order (important if CSV was shuffled)
    df = df.sort_values("Date").reset_index(drop=True)

    # Create a consecutive integer time index (0, 1, 2, ..., n-1)
    # for use as the regression input feature
    t = np.arange(len(df))

    # --- Fit polynomial regression model (degree = 3) on historical data ---

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(t.reshape(-1, 1))

    # Dependent variable: closing prices as a NumPy array
    y = df["Close"].to_numpy()

    model = LinearRegression()
    model.fit(X_poly, y)

    # Model predictions over the historical period
    y_pred = model.predict(X_poly)

    # --- Build future time points and generate extrapolated predictions ---

    last_t = t[-1]
    # Future integer time indices for the next `days_ahead` days
    future_t = np.arange(last_t + 1, last_t + 1 + days_ahead).reshape(-1, 1)
    future_X_poly = poly.transform(future_t)
    future_pred = model.predict(future_X_poly)

    # Build a continuous range of future calendar dates matching `days_ahead`
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=days_ahead,
        freq="D",
    )

    # --- Plot historical data, trend, and future predictions ---

    plt.figure(figsize=(10, 5))

    # Actual closing prices
    plt.plot(df["Date"], df["Close"], label="Actual Price")

    # Polynomial regression trend over historical data
    plt.plot(df["Date"], y_pred, label="Trend", linewidth=2)

    # Extrapolated future predictions (simple and illustrative only)
    plt.plot(
        future_dates,
        future_pred,
        label=f"Future {days_ahead}d Prediction",
        linestyle="--",
        linewidth=2,
    )

    # Labels and title for readability
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"{symbol} Closing Price Over Time")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()