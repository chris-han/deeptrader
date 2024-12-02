import backtrader as bt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Download Data
data = yf.download('OKLO', start='2024-09-01', end='2024-11-27')
df = pd.DataFrame(data)
df.index = pd.to_datetime(df.index)
df.index.name = 'datetime'
# Rename the columns to match CustomPandasData expectations
df.columns = ['close', 'close_no_use', 'high', 'low', 'open', 'volume']

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
        # Prepare the plot
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot the closing prices
        ax.plot(df.index, df['close'], label='Close Price')
        
        # Plot buy signals
        if self.buy_signals:
            buy_dates, buy_prices = zip(*self.buy_signals)
            ax.plot(buy_dates, buy_prices, '^', markersize=10, color='g', label='Buy Signal')
        
        # Plot sell signals
        if self.sell_signals:
            sell_dates, sell_prices = zip(*self.sell_signals)
            ax.plot(sell_dates, sell_prices, 'v', markersize=10, color='r', label='Sell Signal')
        
        # Format x-axis to show dates nicely
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
        
        # Add labels and title
        plt.title('OKLO Stock Price with Buy/Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Add Strategy to Cerebro
cerebro.addstrategy(TestStrategy)

# Run the Backtest
strategies = cerebro.run()

# Note: The plotting is now done within the strategy's stop method