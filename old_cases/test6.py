import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Download Data
data = yf.download('OKLO', start='2024-09-01', end='2024-11-27')
df = pd.DataFrame(data)
df.index = pd.to_datetime(df.index)
df.index.name = 'datetime'
# Rename the columns to match CustomPandasData expectations
df.columns = ['close', 'close_no_use', 'high', 'low', 'open', 'volume']

# Calculate daily log returns
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

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
        self.sma = bt.indicators.TripleExponentialMovingAverage(self.datas[0], period=2)  

    def next(self):
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
        
        # Add labels and title to price subplot
        ax2.set_title('OKLO Stock Price with Buy/Sell Signals')
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