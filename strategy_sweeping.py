# Import necessary libraries
import backtrader as bt
from yahoo_fin.stock_info import get_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from functools import lru_cache
from datetime import datetime

# Cache stock data to reduce fetching time
@lru_cache(maxsize=128)
def get_data_with_cache(ticker, start_date, end_date, index_as_date=True, interval="1d"):
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')
    return get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=index_as_date, interval=interval)

# Example usage
# data = get_data_with_cache('OKLO', '2022-01-01', '2023-01-01')
data = get_data_with_cache('OKLO', '10/01/2024', '12/04/2024')

# Prepare the DataFrame
df = pd.DataFrame(data)
df.index = pd.to_datetime(df.index)
df.index.name = 'datetime'
df.dropna(inplace=True)
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df.dropna(inplace=True)

# Split the DataFrame into training and test sets
train_size = int(len(df) * 0.7)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Function to prepare prediction data
def prepare_prediction_data(df, lookback=5):
    X, y_one_day, y_two_day = [], [], []
    for i in range(lookback, len(df) - 2):
        features = (df['close'].iloc[i-lookback:i].values.tolist() + 
                    df['log_return'].iloc[i-lookback:i].values.tolist())
        X.append(features)
        y_one_day.append(df['close'].iloc[i + 1])
        y_two_day.append(df['close'].iloc[i + 2])
    return np.array(X), np.array(y_one_day), np.array(y_two_day)

# Prepare data and train models
X_train, y_train_one_day, y_train_two_day = prepare_prediction_data(train_df)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
model_one_day = LinearRegression().fit(X_poly_train, y_train_one_day)
model_two_day = LinearRegression().fit(X_poly_train, y_train_two_day)

# Define multiple strategies to sweep
class StrategyTemplate(bt.Strategy):
    params = (
        ('indicator', 'sma'),
        ('period', 14),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        if self.params.indicator == 'sma':
            self.indicator = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.period)
        elif self.params.indicator == 'ema':
            self.indicator = bt.indicators.ExponentialMovingAverage(self.datas[0], period=self.params.period)
        elif self.params.indicator == 'tema':
            self.indicator = bt.indicators.TripleExponentialMovingAverage(self.datas[0], period=self.params.period)

    def next(self):
        if self.dataclose[0] > self.indicator[0]:
            self.buy(size=10)
        elif self.dataclose[0] < self.indicator[0]:
            self.sell(size=10)

# Sweep different indicators and periods
indicators = ['sma', 'ema', 'tema']
periods = [5, 10, 20]

# Store results
results = []

for indicator in indicators:
    for period in periods:
        cerebro = bt.Cerebro()
        train_feed = bt.feeds.PandasData(dataname=train_df)
        test_feed = bt.feeds.PandasData(dataname=test_df)
        cerebro.adddata(train_feed)
        cerebro.adddata(test_feed)
        cerebro.addstrategy(StrategyTemplate, indicator=indicator, period=period)
        cerebro.broker.setcash(100000.0)
        cerebro.broker.setcommission(commission=0.0)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        strat = cerebro.run()[0]
        sharpe = strat.analyzers.sharpe.get_analysis()['sharperatio']
        annual_return = strat.analyzers.annual_return.get_analysis()['rtot']
        drawdown = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
        results.append((indicator, period, sharpe, annual_return, drawdown))

# Print results
print("Results:")
for res in results:
    print(f"Indicator: {res[0]}, Period: {res[1]}, Sharpe Ratio: {res[2]:.4f}, Annual Return: {res[3]:.2f}, Max Drawdown: {res[4]:.2f}")

# Select the best strategy based on Sharpe Ratio
best_strategy = max(results, key=lambda x: x[2])
print("\nBest Strategy based on Sharpe Ratio:")
print(f"Indicator: {best_strategy[0]}, Period: {best_strategy[1]}, Sharpe Ratio: {best_strategy[2]:.4f}")

# Alternatively, you can use drawdown or annual return as criteria:
# best_strategy = min(results, key=lambda x: x[4]) for drawdown (minimize)
# best_strategy = max(results, key=lambda x: x[3]) for annual return (maximize)