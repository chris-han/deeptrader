import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Download Data
data = yf.download('OKLO', start='2024-09-01', end='2024-11-27')
df = pd.DataFrame(data)
df.index = pd.to_datetime(df.index)
df.index.name = 'datetime'
# Rename the columns to match CustomPandasData expectations
df.columns = ['close', 'close_no_use', 'high', 'low', 'open', 'volume']

# Calculate daily log returns
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# Drop rows with NaN values
df = df.dropna()

# Prepare data for prediction
def prepare_prediction_data(df, lookback=5):
    # Create features using lookback periods
    X = []
    y = []
    for i in range(lookback, len(df)):
        # Use close prices and log returns as features
        features = df['close'].iloc[i-lookback:i].values.tolist() + \
                   df['log_return'].iloc[i-lookback:i].values.tolist()
        X.append(features)
        y.append(df['close'].iloc[i])

    return np.array(X), np.array(y)

# Prepare data and train model
X, y = prepare_prediction_data(df)

# Ensure no NaN values in X and y
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
y = y[mask]

# Use polynomial features for non-linear regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train the model
model = LinearRegression()
model.fit(X_poly, y)

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
        self.predict_signals = []  # Store prediction signals
        self.sma = bt.indicators.TripleExponentialMovingAverage(self.datas[0], period=2)

    def next(self):
        # Prepare input for prediction (last 5 days of close and log returns)
        if len(self.data) >= 6:
            # Prepare features for prediction
            last_close = list(self.data.close.get(size=-5))
            last_log_return = list(np.log(np.array(self.data.close.get(size=-5)) / np.array(self.data.close.get(size=-6))))
            
            if len(last_close) == 5 and len(last_log_return) == 5:
                last_features = last_close + last_log_return
                
                # Transform features
                last_features_poly = poly.transform([last_features])
                
                # Predict next day's close price
                predicted_price = model.predict(last_features_poly)[0]
                
                # Prediction signal logic
                if predicted_price > self.dataclose[0]:
                    self.predict_signals.append((self.datas[0].datetime.datetime(0), predicted_price))
                    print(f'Predict Buy: {predicted_price:.2f} at {self.datas[0].datetime.datetime(0)}')
                elif predicted_price < self.dataclose[0]:
                    self.predict_signals.append((self.datas[0].datetime.datetime(0), predicted_price))
                    print(f'Predict Sell: {predicted_price:.2f} at {self.datas[0].datetime.datetime(0)}')

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
        
        # Plot prediction signals in blue
        if self.predict_signals:
            predict_dates, predict_prices = zip(*self.predict_signals)
            ax2.plot(predict_dates, predict_prices, 'o', markersize=10, color='blue', label='Prediction Signal')
            
            # Add price labels for prediction signals
            for date, price in self.predict_signals:
                ax2.annotate(f'${price:.2f}',
                             (date, price),
                             xytext=(10, 0),
                             textcoords='offset points',
                             fontsize=9,
                             color='blue',
                             bbox=dict(boxstyle='round,pad=0.2', fc='lightblue', alpha=0.7))
        
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