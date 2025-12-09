"""
read_and_plot.py
----------------
Command-line script that:

1. Prompts the user for:
   - A stock ticker symbol (e.g., AAPL)
   - A number of future days to predict
2. Downloads 1 year of daily OHLC data for that symbol using yfinance
3. Saves the raw data to a CSV file under the `data/` folder
4. Reads and cleans the saved data
5. Calls `plot_model` to visualize the historical prices and a simple
   polynomial regression-based future projection.

This script is designed as a small, self-contained demo project for
learning data handling, basic modeling, and plotting in Python.
"""

import os

import pandas as pd
import yfinance as yf

from plot_model import plot_model

# Folder where all CSV data files will be stored
FOLDER_NAME = "data"


def get_filepath(symbol):
    """
    Build the full CSV file path for a given stock symbol.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., "AAPL").

    Returns
    -------
    str
        Absolute or relative path to the CSV file within the data folder.
    """
    return os.path.join(FOLDER_NAME, f"{symbol}.csv")


def download_and_save_data(symbol):
    """
    Download 1 year of daily stock price data and save it as CSV.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol to download (e.g., "TSLA", "MSFT").

    Returns
    -------
    bool
        True if data is downloaded and saved successfully.
        False if download fails or returns no data.
    """
    filepath = get_filepath(symbol)

    print(f"--- Step 1: Downloading data for {symbol} ---")

    # Ensure the data folder exists; create it if missing
    os.makedirs(FOLDER_NAME, exist_ok=True)
    print(f"Ensured folder '{FOLDER_NAME}' exists.")

    try:
        # Download up to 1 year of daily prices, auto-adjusted for splits/dividends
        df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True)

        # If no rows were returned, treat it as a failure (bad symbol or network issue)
        if df.empty:
            print(
                f"Error: No data downloaded for {symbol}. "
                "Check the ticker symbol or your network connection."
            )
            return False

        # Move the index (Date) into a regular column
        df = df.reset_index()

        # Save to CSV for later reuse
        df.to_csv(filepath, index=False)
        print(f"Successfully saved data to {filepath}.")

        return True

    except Exception as e:
        # Any unexpected exception (network, I/O, etc.)
        print(f"Error downloading {symbol}: {e}")
        return False


def read_and_process_data(symbol):
    """
    Read the saved CSV for the given symbol and perform basic cleaning.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol whose CSV file should exist in the data folder.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with:
        - 'Date' as datetime
        - 'Close' as numeric (rounded to 2 decimals)
        Rows with missing 'Close' values are dropped.
        An empty DataFrame is returned if the file does not exist.
    """
    filepath = get_filepath(symbol)

    print(f"\n--- Step 2: Reading and processing data from {filepath} ---")

    # If the CSV file isn't found, return an empty DataFrame
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found. Download data first.")
        return pd.DataFrame()

    # Read raw CSV data
    df = pd.read_csv(filepath)

    # Convert the 'Date' column to datetime for proper time-series operations
    df["Date"] = pd.to_datetime(df["Date"])

    # Ensure 'Close' is numeric; coerce invalid entries to NaN, then round
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce").round(2)

    # Drop any rows where 'Close' is missing, then reset index
    df = df.dropna(subset=["Close"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    # --- Command-line interaction and overall script flow ---

    # Ask user for a stock ticker symbol and normalize to uppercase
    symbol = input("Enter stock symbol(e.g. AAPL): ").upper()

    # Step 1: Download and save raw data
    success = download_and_save_data(symbol)

    if success:
        # Step 2: Read back the saved CSV and clean it
        df = read_and_process_data(symbol)

        if not df.empty:
            print(f"Successfully processed {len(df)} trading days of data for {symbol}.")

            # Ask the user how many future days they want to predict
            # Note: This assumes the user enters a valid integer
            days_ahead = int(input("Enter number of days to predict(in days): "))

            # Step 3: Plot historical prices and simple future projection
            plot_model(df, symbol, days_ahead)
        else:
            print(f"Execution halted. Data processing failed for '{symbol}'.")
    else:
        print(f"Execution halted. Download failed for '{symbol}'.")