import backtrader as bt
from yahoo_fin.stock_info import get_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from functools import lru_cache
from datetime import datetime
import gradio as gr
import io

@lru_cache(maxsize=128)
def get_data_with_cache(ticker, start_date, end_date, index_as_date=True, interval="1d"):
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')
    return get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=index_as_date, interval=interval)

# Example usage
data = get_data_with_cache('OKLO', '2022-01-01', '2023-01-01')

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

def backtest_strategy(indicator, period):
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
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio')
    annual_return = strat.analyzers.annual_return.get_analysis().get('rtot')
    drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown')
    return strat, sharpe, annual_return, drawdown

# Sweep different indicators and periods
indicators = ['sma', 'ema', 'tema']
periods = [5, 10, 20]

# Store results
results = []

for indicator in indicators:
    for period in periods:
        strat, sharpe, annual_return, drawdown = backtest_strategy(indicator, period)
        results.append((indicator, period, sharpe, annual_return, drawdown, strat))

# Get best strategy based on Sharpe Ratio
best_strategy = max(results, key=lambda x: x[2])

# Plotting function
def plot_best_strategy(strategy):
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(StrategyTemplate, indicator=best_strategy[0], period=best_strategy[1])
    cerebro.run()
    cerebro.plot[1][0].savefig('best_strategy_plot.png')

def run_backtest():
    plot_best_strategy(best_strategy[5])
    plt.close()
    with open('best_strategy_plot.png', 'rb') as img:
        strategy_plot = img.read()
    metrics = f"Indicator: {best_strategy[0]}, Period: {best_strategy[1]}\nSharpe Ratio: {best_strategy[2]:.4f}\nAnnual Return: {best_strategy[3]:.2f}\nMax Drawdown: {best_strategy[4]:.2f}"
    return strategy_plot, metrics

def main():
    interface = gr.Interface(
        fn=run_backtest,
        inputs=[],
        outputs=[
            gr.outputs.Image(type="auto", label="Best Strategy Plot"),
            gr.outputs.Textbox(label="Metrics Comparison")
        ]
    )
    interface.launch()

if __name__ == "__main__":
    main()