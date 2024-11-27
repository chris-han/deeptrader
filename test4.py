import backtrader as bt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
# Download Data
data = yf.download('OKLO', start='2024-09-01', end='2024-11-27')
df = pd.DataFrame(data)
# Rename the columns to match CustomPandasData expectations
df.columns = ['adj_close', 'close', 'high', 'low', 'open', 'volume']

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
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=20)  # 20-day moving average

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
        ax = plt.gca()  # Get the current Axes instance on the current figure
        if self.buy_signals:
            buy_dates, buy_prices = zip(*self.buy_signals)
            ax.plot(buy_dates, buy_prices, '^', markersize=10, color='g', label='Buy Signal')
        if self.sell_signals:
            sell_dates, sell_prices = zip(*self.sell_signals)
            ax.plot(sell_dates, sell_prices, 'v', markersize=10, color='r', label='Sell Signal')
        ax.legend()

# Add Strategy to Cerebro
cerebro.addstrategy(TestStrategy)

# Run the Backtest
strategies = cerebro.run()

# Plot the result with matplotlib
fig, ax = plt.subplots(figsize=(14, 7))
cerebro.plot(volume=True, ax=ax)  # Disable volume plotting for clarity

