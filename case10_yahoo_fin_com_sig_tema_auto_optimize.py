import backtrader as bt
from yahoo_fin.stock_info import get_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from functools import lru_cache
from datetime import datetime

@lru_cache(maxsize=128)
def get_data_with_cache(ticker, start_date, end_date, index_as_date=True, interval="1d"):
    if isinstance(start_date, datetime):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, datetime):
        end_date = end_date.strftime('%Y-%m-%d')
    return get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=index_as_date, interval=interval)

def prepare_prediction_data(df, lookback=5):
    """Prepares data for prediction models."""
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    
    X = []
    y_one_day = []
    y_two_day = []
    y_three_day = []
    y_four_day = []
    
    for i in range(lookback, len(df) - 4):
        close_prices = df['close'].iloc[i-lookback:i].values
        log_return_history = log_returns.iloc[i-lookback:i].values

        features = np.concatenate([close_prices, log_return_history])
        X.append(features)
        y_one_day.append(df['close'].iloc[i + 1])
        y_two_day.append(df['close'].iloc[i + 2])
        y_three_day.append(df['close'].iloc[i + 3])
        y_four_day.append(df['close'].iloc[i + 4])

    return np.array(X), np.array(y_one_day), np.array(y_two_day), np.array(y_three_day), np.array(y_four_day)

class TestStrategy(bt.Strategy):
    params = (
        ('period', 6),
        ('optimize', False),
    )

    LONGEST_PERIOD = 6
    # Class variables to store data
    df = None
    X = None
    y_one_day = None
    y_two_day = None
    y_three_day = None
    y_four_day = None

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.buy_signals = []
        self.sell_signals = []
        self.predict_signals_one_day = []
        self.predict_signals_two_day = []
        self.predict_signals_three_day = []
        self.predict_signals_four_day = []
        self.tema = bt.indicators.TripleExponentialMovingAverage(
            self.datas[0], period=self.params.period)
        self.last_dataset_date = self.df.index[-1]

        self.poly = PolynomialFeatures(degree=2)
        X_poly = self.poly.fit_transform(self.X)

        self.model_one_day = LinearRegression()
        self.model_one_day.fit(X_poly, self.y_one_day)

        self.model_two_day = LinearRegression()
        self.model_two_day.fit(X_poly, self.y_two_day)

        self.model_three_day = LinearRegression()
        self.model_three_day.fit(X_poly, self.y_three_day)

        self.model_four_day = LinearRegression()
        self.model_four_day.fit(X_poly, self.y_four_day)

    def next(self):
        if len(self.data) >= 6:
            try:
                last_close_prices = [self.data.close[-i] for i in range(1, 6)]
                last_close_prices.reverse()

                last_log_returns = [
                    np.log(self.data.close[-i] / self.data.close[-i-1]) 
                    for i in range(1, 6)
                ]
                last_log_returns.reverse()

                last_features = last_close_prices + last_log_returns
                last_features_poly = self.poly.transform([last_features])
                
                predicted_price_one_day = self.model_one_day.predict(last_features_poly)[0]
                predicted_price_two_day = self.model_two_day.predict(last_features_poly)[0]
                predicted_price_three_day = self.model_three_day.predict(last_features_poly)[0]
                predicted_price_four_day = self.model_four_day.predict(last_features_poly)[0]
                
                current_date = self.datas[0].datetime.datetime(0)
                if current_date.date() == self.last_dataset_date.date():
                    next_day = current_date + pd.Timedelta(days=1)
                    two_days_later = current_date + pd.Timedelta(days=2)
                    three_days_later = current_date + pd.Timedelta(days=3)
                    four_days_later = current_date + pd.Timedelta(days=4)
                    self.predict_signals_one_day.append((next_day, predicted_price_one_day))
                    self.predict_signals_two_day.append((two_days_later, predicted_price_two_day))
                    self.predict_signals_three_day.append((three_days_later, predicted_price_three_day))
                    self.predict_signals_four_day.append((four_days_later, predicted_price_four_day))
                    print(f'Predict Price for Next Day: {predicted_price_one_day:.2f} at {next_day}')
                    print(f'Predict Price for Two Days Later: {predicted_price_two_day:.2f} at {two_days_later}')
                    print(f'Predict Price for Three Days Later: {predicted_price_three_day:.2f} at {three_days_later}')
                    print(f'Predict Price for Four Days Later: {predicted_price_four_day:.2f} at {four_days_later}')

            except Exception as e:
                print(f"Prediction error: {e}")

        if self.data.close > self.tema:
            self.buy_signals.append((self.datas[0].datetime.datetime(0), self.dataclose[0]))
            self.buy(size=10)
            print(f'Buy signal: {self.dataclose[0]} at {self.datas[0].datetime.datetime(0)}')
        elif self.data.close < self.tema:
            self.sell_signals.append((self.datas[0].datetime.datetime(0), self.dataclose[0]))
            self.sell(size=10)
            print(f'Sell signal: {self.dataclose[0]} at {self.datas[0].datetime.datetime(0)}')

    def notify_trade(self, trade):
        if trade.isclosed:
            return

    def _plot_strategy(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 2, 1]}, sharex=True)
        
        ax1.bar(self.df.index[1:], self.df['log_return'][1:], color='blue', alpha=0.6, label='Daily Log Return')
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_title('Daily Log Returns')
        ax1.set_ylabel('Log Return')
        ax1.legend()
        ax1.grid(True)
        
        cumulative_mean = self.df['log_return'][1:].cumsum()
        ax1_twin = ax1.twinx()
        ax1_twin.plot(self.df.index[1:], cumulative_mean, color='green', label='Cumulative Mean')
        ax1_twin.set_ylabel('Cumulative Mean Log Return')
        ax1_twin.legend(loc='lower right')
        
        ax2.plot(self.df.index, self.df['close'], label='Close Price')
        
        def annotate_signals(ax, signals, marker, color, label, text_offset):
           if signals:
                signal_dates, signal_prices = zip(*signals)
                ax.plot(signal_dates, signal_prices, marker, markersize=10, color=color, label=label)
                for date, price in signals:
                    ax.annotate(f'${price:.2f}', 
                                (date, price), 
                                xytext=text_offset,  
                                textcoords='offset points', 
                                fontsize=9, 
                                color=color,
                             bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
            
        annotate_signals(ax2, self.buy_signals, '^', 'g', 'Buy Signal', (10, 10))
        annotate_signals(ax2, self.sell_signals, 'v', 'r', 'Sell Signal', (10, -15))
        annotate_signals(ax2, self.predict_signals_one_day, 'o', 'blue', 'One-Day Prediction Signal', (10, 0))
        annotate_signals(ax2, self.predict_signals_two_day, 'o', 'cyan', 'Two-Day Prediction Signal', (10, 0))
        annotate_signals(ax2, self.predict_signals_three_day, 'o', 'magenta', 'Three-Day Prediction Signal', (10, 0))
        annotate_signals(ax2, self.predict_signals_four_day, 'o', 'orange', 'Four-Day Prediction Signal', (10, 0))
        
        def add_prediction_markers(ax, signals, color):
            if signals:
                signal_dates, _ = zip(*signals)
                for date in signal_dates:
                    ax.axvline(x=date, color=color, linestyle='--', lw=2)
                    
        add_prediction_markers(ax2, self.predict_signals_one_day, 'blue')
        add_prediction_markers(ax2, self.predict_signals_two_day, 'cyan')
        add_prediction_markers(ax2, self.predict_signals_three_day, 'magenta')
        add_prediction_markers(ax2, self.predict_signals_four_day, 'orange')
        ax2.set_title('Stock Price with Signals')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True)
        
        ax3.bar(self.df.index, self.df['volume'], color='gray', alpha=0.6, label='Volume')
        ax3.set_ylabel('Volume')
        ax3.legend()
        ax3.grid(True)
        
        plt.gcf().autofmt_xdate()
        
        next_day = self.last_dataset_date + pd.Timedelta(days=1)
        ax3.axvline(x=next_day, color='blue', linestyle=':', lw=2)
        two_days_later = self.last_dataset_date + pd.Timedelta(days=2)
        ax3.axvline(x=two_days_later, color='cyan', linestyle=':', lw=2)
        three_days_later = self.last_dataset_date + pd.Timedelta(days=3)
        ax3.axvline(x=three_days_later, color='magenta', linestyle=':', lw=2)
        four_days_later = self.last_dataset_date + pd.Timedelta(days=4)
        ax3.axvline(x=four_days_later, color='orange', linestyle=':', lw=2)

        print("Log Return Statistics:")
        print(f"Mean Daily Log Return: {self.df['log_return'][1:].mean():.4f}")
        print(f"Standard Deviation of Log Returns: {self.df['log_return'][1:].std():.4f}")
        print(f"Cumulative Log Return: {self.df['log_return'][1:].sum():.4f}")
        
        plt.tight_layout()
        plt.show()

    def stop(self):
        if self.params.optimize:
            return
        
        self._plot_strategy()


def optimize_period(df):
    """Optimizes the TEMA period."""
    best_period = None
    best_performance = float('-inf')

    if len(df) < TestStrategy.LONGEST_PERIOD:
        raise ValueError("Insufficient data length for optimization")

    # Prepare prediction data for optimization
    X, y_one_day, y_two_day, y_three_day, y_four_day = prepare_prediction_data(df)

    data_feed = bt.feeds.PandasData(dataname=df)
    periods_to_test = range(1, min(len(df) // 2, TestStrategy.LONGEST_PERIOD + 1))

    for period in periods_to_test:
        cerebro = bt.Cerebro()
        cerebro.adddata(data_feed)
        # Add strategy with all required parameters
        cerebro.addstrategy(TestStrategy, 
                           period=period, 
                           optimize=True)
        
        # Store the data as class variables
        TestStrategy.df = df
        TestStrategy.X = X
        TestStrategy.y_one_day = y_one_day
        TestStrategy.y_two_day = y_two_day
        TestStrategy.y_three_day = y_three_day
        TestStrategy.y_four_day = y_four_day
        
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        result = cerebro.run()
        if result:
            performance = result[0].analyzers.returns.get_analysis()['rnorm']
        else:
            performance = float('-inf')
            
        if performance > best_performance:
            best_performance = performance
            best_period = period

    return best_period


if __name__ == '__main__':
    data = get_data_with_cache('OKLO', '10/01/2024', '12/06/2024')
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'datetime'
    df.dropna(inplace=True)

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)
    X, y_one_day, y_two_day, y_three_day, y_four_day = prepare_prediction_data(df)

    # Set class variables
    TestStrategy.df = df
    TestStrategy.X = X
    TestStrategy.y_one_day = y_one_day
    TestStrategy.y_two_day = y_two_day
    TestStrategy.y_three_day = y_three_day
    TestStrategy.y_four_day = y_four_day

    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(dataname=df)

    best_period = optimize_period(df)
    print(f"Best period found: {best_period}")

    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(TestStrategy, period=best_period)
    result = cerebro.run()
