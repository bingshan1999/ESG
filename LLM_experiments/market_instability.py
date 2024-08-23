import yfinance as yf
import pandas as pd
from datetime import datetime

# Define the tickers for Bitcoin, Solana, Ripple, and Dogecoin
tickers = ['BTC-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD']

# Download historical price data for the last 3 years
end_date = datetime.now()
start_date = end_date - pd.DateOffset(years=3)
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Resample data to get the closing price at the end of each quarter
quarterly_data = data.resample('QE').last()

# Calculate quarterly percentage changes
quarterly_changes = quarterly_data.pct_change() * 100

# Define the thresholds based on the sentences
thresholds = pd.Series({'BTC-USD': -14, 'SOL-USD': -30, 'XRP-USD': -23, 'DOGE-USD': -42})

# Align the quarterly changes DataFrame with the thresholds Series
aligned_changes, aligned_thresholds = quarterly_changes.align(thresholds, axis=1)

# Check if the price drop in each quarter is equal to or greater than the thresholds
significant_drops = (aligned_changes <= aligned_thresholds).sum()

# Display the results
print("Frequency of quarters with significant drops:")
print(significant_drops)

# Display the actual quarterly percentage drops for comparison
print("\nQuarterly percentage drops for the last 3 years:")
print(quarterly_changes)
