import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import yfinance as yf
import os 
from plot_model import plot_model

FOLDER_NAME = "data"

def get_filepath(symbol):
    file_name = f"{symbol}.csv"
    return os.path.join(FOLDER_NAME, file_name)
    
def download_and_save_data(symbol):
    filepath = get_filepath(symbol)
    
    print(f"--- Step 1: Downloading data for {symbol} ---")
    
    os.makedirs(FOLDER_NAME, exist_ok=True)
    print(f"Ensured folder '{FOLDER_NAME}' exists.")
    
    df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True)
    df = df.reset_index()

    df.to_csv(filepath, index=False)
    
    print(f"Successfully saved data to {filepath}.")
    
    return True

def read_and_process_data(symbol):
    filepath = get_filepath(symbol)
    
    print(f"\n--- Step 2: Reading and Processing data from {filepath} ---")
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found. Please ensure data was downloaded first.")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Close"] = df["Close"].round(2)
    
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    mySymbol = input("Enter stock symbol name (eg: AAPL): ").upper()
    success = download_and_save_data(mySymbol)
    
    if success:
        df = read_and_process_data(mySymbol)
        
        if not df.empty:
            print(f"Successfully processed {len(df)} trading days of data for {mySymbol}.")
            plot_model(df, mySymbol, 30)
        else:
            print(f"Execution halted. Data processing failed for '{mySymbol}'.")