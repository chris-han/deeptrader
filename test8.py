import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# from sklearn.impute import SimpleImputer

# Download Data
data = yf.download('OKLO', start='2024-10-01', end='2024-11-29')
df = pd.DataFrame(data)
df.index = pd.to_datetime(df.index)
df.index.name = 'datetime'
# Rename the columns to match CustomPandasData expectations
df.columns = ['close', 'close_no_use', 'high', 'low', 'open', 'volume']

# Clean and prepare data
df.dropna(inplace=True)  # Remove rows with NaN values

# Calculate daily log returns
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df.dropna(inplace=True)  # Remove NaN values after log return calculation

# Prepare data for prediction
def prepare_prediction_data(df, lookback=5):
    # Create features using lookback periods
    X = []
    y_one_day = []
    y_two_day = []
    y_three_day = []
    for i in range(lookback, len(df) - 3):
        # Use close prices and log returns as features
        features = (
            df['close'].iloc[i-lookback:i].values.tolist() + 
            df['log_return'].iloc[i-lookback:i].values.tolist()
        )
        X.append(features)
        y_one_day.append(df['close'].iloc[i + 1])  # Shift by 1 for one-day prediction
        y_two_day.append(df['close'].iloc[i + 2])  # Shift by 2 for two-day prediction
        y_three_day.append(df['close'].iloc[i + 3])  # Shift by 3 for three-day prediction
    
    return np.array(X), np.array(y_one_day), np.array(y_two_day), np.array(y_three_day)

# Prepare data and train models
X, y_one_day, y_two_day, y_three_day = prepare_prediction_data(df)

# Use polynomial features for non-linear regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train the models
model_one_day = LinearRegression()
model_one_day.fit(X_poly, y_one_day)

model_two_day = LinearRegression()
model_two_day.fit(X_poly, y_two_day)

model_three_day = LinearRegression()
model_three_day.fit(X_poly, y_three_day)

# Create a Cerebro Engine
cerebro = bt.Cerebro()

# Add Data Feed to Cerebro
data_feed = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data_feed)

# Define a Simple Strategy with buy and sell signal recording
class TestStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.buy_signals = []  # Store the buy signal dates
        self.sell_signals = []  # Store the sell signal dates
        self.predict_signals_one_day = []  # Store one-day prediction signals
        self.predict_signals_two_day = []  # Store two-day prediction signals
        self.predict_signals_three_day = []  # Store three-day prediction signals
        self.sma = bt.indicators.TripleExponentialMovingAverage(self.datas[0], period=2)
        
        # Store the last date in the dataset for prediction timing
        self.last_dataset_date = df.index[-1]

    def next(self):
        # Ensure we have enough data for prediction
        if len(self.data) >= 6:
            try:
                # Extract last 5 closing prices
                last_close_prices = [self.data.close[-i] for i in range(1, 6)]
                last_close_prices.reverse()  # Ensure chronological order

                # Extract last 5 log returns
                last_log_returns = [
                    np.log(self.data.close[-i] / self.data.close[-i-1]) 
                    for i in range(1, 6)
                ]
                last_log_returns.reverse()  # Ensure chronological order

                # Combine features
                last_features = last_close_prices + last_log_returns
                
                # Transform features
                last_features_poly = poly.transform([last_features])
                
                # Predict next day's, two day's and three day's close price
                predicted_price_one_day = model_one_day.predict(last_features_poly)[0]
                predicted_price_two_day = model_two_day.predict(last_features_poly)[0]
                predicted_price_three_day = model_three_day.predict(last_features_poly)[0]
                
                # Only show predictions for the day after the last dataset date
                current_date = self.datas[0].datetime.datetime(0)
                if current_date.date() == self.last_dataset_date.date():
                    next_day = current_date + pd.Timedelta(days=1)
                    two_days_later = current_date + pd.Timedelta(days=2)
                    three_days_later = current_date + pd.Timedelta(days=3)
                    self.predict_signals_one_day.append((next_day, predicted_price_one_day))
                    self.predict_signals_two_day.append((two_days_later, predicted_price_two_day))
                    self.predict_signals_three_day.append((three_days_later, predicted_price_three_day))
                    print(f'Predict Price for Next Day: {predicted_price_one_day:.2f} at {next_day}')
                    print(f'Predict Price for Two Days Later: {predicted_price_two_day:.2f} at {two_days_later}')
                    print(f'Predict Price for Three Days Later: {predicted_price_three_day:.2f} at {three_days_later}')

            except Exception as e:
                print(f"Prediction error: {e}")

        # Original buy/sell strategy
        if self.dataclose[0] > self.dataclose[-1]:
            self.buy_signals.append((self.datas[0].datetime.datetime(0), self.dataclose[0]))
            self.buy(size=10)  # Buying 10 shares
            print(f'Buy signal: {self.dataclose[0]} at {self.datas[0].datetime.datetime(0)}')

        elif self.dataclose[0] < self.dataclose[-1]:
            self.sell_signals.append((self.datas[0].datetime.datetime(0), self.dataclose[0]))
            self.sell(size=10)  # Selling 10 shares
            print(f'Sell signal: {self.dataclose[0]} at {self.datas[0].datetime.datetime(0)}')

    def notify_trade(self, trade):
        if trade.isclosed:
            return

    def stop(self):
        # Create a figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), 
                                            gridspec_kw={'height_ratios': [1, 2, 1]}, 
                                            sharex=True)
        
        # Plot daily log returns on the top subplot
        ax1.bar(df.index[1:], df['log_return'][1:], color='blue', alpha=0.6, label='Daily Log Return')
        ax1.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at zero
        ax1.set_title('Daily Log Returns')
        ax1.set_ylabel('Log Return')
        ax1.legend()
        ax1.grid(True)
        
        # Add cumulative mean line
        cumulative_mean = df['log_return'][1:].cumsum()
        ax1_twin = ax1.twinx()
        ax1_twin.plot(df.index[1:], cumulative_mean, color='green', label='Cumulative Mean')
        ax1_twin.set_ylabel('Cumulative Mean Log Return')
        ax1_twin.legend(loc='lower right')
        
        # Plot the closing prices on the middle subplot
        ax2.plot(df.index, df['close'], label='Close Price')
        
        # Plot buy signals with price labels
        if self.buy_signals:
            buy_dates, buy_prices = zip(*self.buy_signals)
            ax2.plot(buy_dates, buy_prices, '^', markersize=10, color='g', label='Buy Signal')
            
            # Add price labels for buy signals
            for date, price in self.buy_signals:
                ax2.annotate(f'${price:.2f}', 
                            (date, price), 
                            xytext=(10, 10),  
                            textcoords='offset points', 
                            fontsize=9, 
                            color='g',
                            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
        
        # Plot sell signals with price labels
        if self.sell_signals:
            sell_dates, sell_prices = zip(*self.sell_signals)
            ax2.plot(sell_dates, sell_prices, 'v', markersize=10, color='r', label='Sell Signal')
            
            # Add price labels for sell signals
            for date, price in self.sell_signals:
                ax2.annotate(f'${price:.2f}', 
                            (date, price), 
                            xytext=(10, -15),  
                            textcoords='offset points', 
                            fontsize=9, 
                            color='r',
                            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
        
        # Plot prediction signals for one-day, two-day, and three-day in different colors
        if self.predict_signals_one_day:
            predict_dates_one_day, predict_prices_one_day = zip(*self.predict_signals_one_day)
            ax2.plot(predict_dates_one_day, predict_prices_one_day, 'o', markersize=10, color='blue', label='One-Day Prediction Signal')

            # Add price labels and vertical lines for one-day prediction signals
            for date, price in self.predict_signals_one_day:
                ax2.annotate(f'${price:.2f}', 
                            (date, price), 
                            xytext=(10, 0),  
                            textcoords='offset points', 
                            fontsize=9, 
                            color='blue',
                            bbox=dict(boxstyle='round,pad=0.2', fc='lightblue', alpha=0.7))
                ax2.axvline(x=date, color='blue', linestyle='--', lw=2)  # Add vertical line

        if self.predict_signals_two_day:
            predict_dates_two_day, predict_prices_two_day = zip(*self.predict_signals_two_day)
            ax2.plot(predict_dates_two_day, predict_prices_two_day, 'o', markersize=10, color='cyan', label='Two-Day Prediction Signal')

            # Add price labels and vertical lines for two-day prediction signals
            for date, price in self.predict_signals_two_day:
                ax2.annotate(f'${price:.2f}', 
                            (date, price), 
                            xytext=(10, 0),  
                            textcoords='offset points', 
                            fontsize=9, 
                            color='cyan',
                            bbox=dict(boxstyle='round,pad=0.2', fc='lightcyan', alpha=0.7))
                ax2.axvline(x=date, color='cyan', linestyle='--', lw=2)  # Add vertical line

        if self.predict_signals_three_day:
            predict_dates_three_day, predict_prices_three_day = zip(*self.predict_signals_three_day)
            ax2.plot(predict_dates_three_day, predict_prices_three_day, 'o', markersize=10, color='magenta', label='Three-Day Prediction Signal')

            # Add price labels and vertical lines for three-day prediction signals
            for date, price in self.predict_signals_three_day:
                ax2.annotate(f'${price:.2f}', 
                            (date, price), 
                            xytext=(10, 0),  
                            textcoords='offset points', 
                            fontsize=9, 
                            color='magenta',
                            bbox=dict(boxstyle='round,pad=0.2', fc='lightpink', alpha=0.7))
                ax2.axvline(x=date, color='magenta', linestyle='--', lw=2)  # Add vertical line

        # Add labels and title to price subplot
        ax2.set_title('OKLO Stock Price with Signals')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True)
        
        # Plot volume on the bottom subplot
        ax3.bar(df.index, df['volume'], color='gray', alpha=0.6, label='Volume')
        ax3.set_ylabel('Volume')
        ax3.legend()
        ax3.grid(True)
        
        # Format x-axis to show dates nicely
        plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
        
        # Add next dates on x-axis in blue
        next_day = self.last_dataset_date + pd.Timedelta(days=1)
        ax3.axvline(x=next_day, color='blue', linestyle=':', lw=2)
        two_days_later = self.last_dataset_date + pd.Timedelta(days=2)
        ax3.axvline(x=two_days_later, color='cyan', linestyle=':', lw=2)
        three_days_later = self.last_dataset_date + pd.Timedelta(days=3)
        ax3.axvline(x=three_days_later, color='magenta', linestyle=':', lw=2)

        # Additional log return statistics
        print("Log Return Statistics:")
        print(f"Mean Daily Log Return: {df['log_return'][1:].mean():.4f}")
        print(f"Standard Deviation of Log Returns: {df['log_return'][1:].std():.4f}")
        print(f"Cumulative Log Return: {df['log_return'][1:].sum():.4f}")
        
        plt.tight_layout()
        plt.show()

# Add Strategy to Cerebro
cerebro.addstrategy(TestStrategy)

# Run the Backtest
strategies = cerebro.run()