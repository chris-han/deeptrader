# \home\chris\repo\deeptrader\test2.py
import backtrader as bt
import yfinance as yf
import pandas as pd

# Download Data
data = yf.download('AAPL', start='2020-01-01', end='2021-01-01')
df = pd.DataFrame(data)
# Rename the columns to match CustomPandasData expectations
df.columns = ['adj_close', 'close', 'high', 'low', 'open', 'volume']

# Create a Cerebro Engine
cerebro = bt.Cerebro()

# Add Data Feed to Cerebro
data_feed = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data_feed)

# Define a Simple Strategy with buy signal recording
class TestStrategy(bt.Strategy):
    params = (('period', 20),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.buy_signal = bt.indicators.CrossOver(self.dataclose, self.dataclose(-1))

    def next(self):
        if self.buy_signal > 0:
            self.buy(size=10)  # Buying 10 shares
            print(f'Buy signal: {self.dataclose[0]}')

# Add Strategy to Cerebro
cerebro.addstrategy(TestStrategy)

# Run the Backtest
strategies = cerebro.run()

# Plot the result
cerebro.plot()