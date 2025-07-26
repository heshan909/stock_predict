#stock_predict.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

### Fetch historical stock data with error handling ###

def fetch_stock_data(ticker, start_date, end_date):  
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

### Prepare data for modeling with date preservation ###

def prepare_data(data, look_back=60):
    df = data[['Close']].copy()
    df = df.reset_index()  # Convert index to column
    dates = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Remove timezone
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    
    # Create sequences with corresponding dates
    X, y, seq_dates = [], [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
        seq_dates.append(dates[i])  # Store date for each prediction point
    
    # Split data
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates_test = seq_dates[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler, df, dates_test

### Train linear regression model ###

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

### Generate predictions from trained model ###

def make_predictions(model, X_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    return predictions

### Generate comprehensive summary DataFrame ###

def generate_summary(data, predictions, dates_test, ticker):
    
    # Create DataFrames for historical and predicted data
    history_df = data[['Date', 'Close']].copy()
    history_df['Date'] = pd.to_datetime(history_df['Date']).dt.tz_localize(None)
    history_df.rename(columns={'Close': 'Actual_Price'}, inplace=True)
    
    preds_df = pd.DataFrame({
        'Date': pd.to_datetime(dates_test),
        'Predicted_Price': predictions.flatten()
    })
    
    # Merge and calculate metrics
    summary_df = pd.merge(history_df, preds_df, on='Date', how='left')
    summary_df['Residual'] = summary_df['Actual_Price'] - summary_df['Predicted_Price']
    
    # Add moving averages
    summary_df['30D_MA'] = summary_df['Actual_Price'].rolling(30, min_periods=1).mean()
    summary_df['60D_MA'] = summary_df['Actual_Price'].rolling(60, min_periods=1).mean()
    
    # Calculate daily returns
    summary_df['Daily_Return'] = summary_df['Actual_Price'].pct_change() * 100
    
    # Add prediction error percentage
    mask = ~summary_df['Predicted_Price'].isna()
    summary_df.loc[mask, 'Error_Pct'] = (abs(summary_df.loc[mask, 'Residual']) / 
                                       summary_df.loc[mask, 'Actual_Price']) * 100
    
    return summary_df.sort_values('Date')

### Save all summaries to a single Excel file with separate sheets ###
def save_to_excel(tickers_summaries, filename="stock_analysis_summary.xlsx"):
    with pd.ExcelWriter(filename) as writer:
        for ticker, summary_df in tickers_summaries:
            summary_df.to_excel(writer, sheet_name=ticker[:31], index=False)
    print(f"\nAll summaries saved to {filename}")

### Create a single plot showing all stock prices ###

def plot_all_prices(tickers_summaries):
    plt.figure(figsize=(15, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(tickers_summaries)))
    
    for idx, (ticker, summary_df) in enumerate(tickers_summaries):
        plt.plot(summary_df['Date'], summary_df['Actual_Price'], 
                label=ticker, color=colors[idx], linewidth=2)
    
    plt.title('Stock Prices Comparison', fontsize=16, pad=20)
    plt.ylabel('Price ($)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('all_stock_prices.png', bbox_inches='tight', dpi=300)
    plt.show()

### Create a single plot showing all daily returns ###

def plot_all_returns(tickers_summaries):
    
    plt.figure(figsize=(15, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(tickers_summaries)))
    
    for idx, (ticker, summary_df) in enumerate(tickers_summaries):
        plt.plot(summary_df['Date'], summary_df['Daily_Return'], 
                label=ticker, color=colors[idx], alpha=0.8, linewidth=1.5)
    
    plt.title('Daily Returns Comparison', fontsize=16, pad=20)
    plt.ylabel('Daily Return (%)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('all_daily_returns.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Configuration
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "PEP"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    look_back = 60
    
    tickers_summaries = []
    
    print("Starting stock analysis...")
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        
        # Fetch data
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is None:
            continue
            
        # Prepare data
        X_train, X_test, y_train, y_test, scaler, processed_data, dates_test = prepare_data(data, look_back)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Make predictions
        predictions = make_predictions(model, X_test, scaler)
        
        # Generate summary
        summary_df = generate_summary(processed_data, predictions, dates_test, ticker)
        tickers_summaries.append((ticker, summary_df))
        
        # Show sample
        print(f"\n{ticker} last 5 days:")
        print(summary_df.tail().to_string(index=False))
    
    # Save and plot results
    if tickers_summaries:
        save_to_excel(tickers_summaries)
        plot_all_prices(tickers_summaries)
        plot_all_returns(tickers_summaries)
        print("\nAnalysis complete! Check the Excel file and PNG images.")
    else:
        print("\nNo data was processed. Please check your ticker symbols.")