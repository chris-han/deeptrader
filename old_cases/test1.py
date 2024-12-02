import backtrader as bt
import yfinance as yf
import pandas as pd
# Download Data
data = yf.download('AAPL', start='2020-01-01', end='2021-01-01')
df = pd.DataFrame(data)
# print(df)
# Rename the columns to match CustomPandasData expectations
df.columns = ['adj_close', 'close', 'high', 'low', 'open', 'volume']
# print(df)

# Create a Cerebro Engine
cerebro = bt.Cerebro()

# Add Data Feed to Cerebro
data_feed = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data_feed)
type(data_feed)
# Define a Simple Strategy
class TestStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close

    def next(self):
        if self.dataclose[0] > self.dataclose[-1]:
            print(f'Buy signal: {self.dataclose[0]}')

# Add Strategy to Cerebro
cerebro.addstrategy(TestStrategy)

# Run the Backtest
cerebro.run()
cerebro.plot()