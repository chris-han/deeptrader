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
    X = []
    y_one_day = []
    y_two_day = []
    y_three_day = []
    y_four_day = []
    
    for i in range(lookback, len(df) - 4):
        features = (
            df['close'].iloc[i-lookback:i].values.tolist() + 
            df['log_return'].iloc[i-lookback:i].values.tolist()
        )
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
        self.last_dataset_date = TestStrategy.df.index[-1]

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
                last_features_poly = TestStrategy.poly.transform([last_features])
                
                predicted_price_one_day = TestStrategy.model_one_day.predict(last_features_poly)[0]
                predicted_price_two_day = TestStrategy.model_two_day.predict(last_features_poly)[0]
                predicted_price_three_day = TestStrategy.model_three_day.predict(last_features_poly)[0]
                predicted_price_four_day = TestStrategy.model_four_day.predict(last_features_poly)[0]
                
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

    def stop(self):
        if self.params.optimize:
            return  # Skip plotting when optimizing periods
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 2, 1]}, sharex=True)
        
        ax1.bar(TestStrategy.df.index[1:], TestStrategy.df['log_return'][1:], color='blue', alpha=0.6, label='Daily Log Return')
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_title('Daily Log Returns')
        ax1.set_ylabel('Log Return')
        ax1.legend()
        ax1.grid(True)
        
        cumulative_mean = TestStrategy.df['log_return'][1:].cumsum()
        ax1_twin = ax1.twinx()
        ax1_twin.plot(TestStrategy.df.index[1:], cumulative_mean, color='green', label='Cumulative Mean')
        ax1_twin.set_ylabel('Cumulative Mean Log Return')
        ax1_twin.legend(loc='lower right')
        
        ax2.plot(TestStrategy.df.index, TestStrategy.df['close'], label='Close Price')
        
        if self.buy_signals:
            buy_dates, buy_prices = zip(*self.buy_signals)
            ax2.plot(buy_dates, buy_prices, '^', markersize=10, color='g', label='Buy Signal')
            for date, price in self.buy_signals:
                ax2.annotate(f'${price:.2f}', 
                             (date, price), 
                             xytext=(10, 10),  
                             textcoords='offset points', 
                             fontsize=9, 
                             color='g',
                             bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
            
        if self.sell_signals:
            sell_dates, sell_prices = zip(*self.sell_signals)
            ax2.plot(sell_dates, sell_prices, 'v', markersize=10, color='r', label='Sell Signal')
            for date, price in self.sell_signals:
                ax2.annotate(f'${price:.2f}', 
                             (date, price), 
                             xytext=(10, -15),  
                             textcoords='offset points', 
                             fontsize=9, 
                             color='r',
                             bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
        
        if self.predict_signals_one_day:
            predict_dates_one_day, predict_prices_one_day = zip(*self.predict_signals_one_day)
            ax2.plot(predict_dates_one_day, predict_prices_one_day, 'o', markersize=10, color='blue', label='One-Day Prediction Signal')
            for date, price in self.predict_signals_one_day:
                ax2.annotate(f'${price:.2f}', 
                             (date, price), 
                             xytext=(10, 0),  
                             textcoords='offset points', 
                             fontsize=9, 
                             color='blue',
                             bbox=dict(boxstyle='round,pad=0.2', fc='lightblue', alpha=0.7))
                ax2.axvline(x=date, color='blue', linestyle='--', lw=2)
        
        if self.predict_signals_two_day:
            predict_dates_two_day, predict_prices_two_day = zip(*self.predict_signals_two_day)
            ax2.plot(predict_dates_two_day, predict_prices_two_day, 'o', markersize=10, color='cyan', label='Two-Day Prediction Signal')
            for date, price in self.predict_signals_two_day:
                ax2.annotate(f'${price:.2f}', 
                             (date, price), 
                             xytext=(10, 0),  
                             textcoords='offset points', 
                             fontsize=9, 
                             color='cyan',
                             bbox=dict(boxstyle='round,pad=0.2', fc='lightcyan', alpha=0.7))
                ax2.axvline(x=date, color='cyan', linestyle='--', lw=2)
        
        if self.predict_signals_three_day:
            predict_dates_three_day, predict_prices_three_day = zip(*self.predict_signals_three_day)
            ax2.plot(predict_dates_three_day, predict_prices_three_day, 'o', markersize=10, color='magenta', label='Three-Day Prediction Signal')
            for date, price in self.predict_signals_three_day:
                ax2.annotate(f'${price:.2f}', 
                             (date, price), 
                             xytext=(10, 0),  
                             textcoords='offset points', 
                             fontsize=9, 
                             color='magenta',
                             bbox=dict(boxstyle='round,pad=0.2', fc='lightpink', alpha=0.7))
                ax2.axvline(x=date, color='magenta', linestyle='--', lw=2)
        
        if self.predict_signals_four_day:
            predict_dates_four_day, predict_prices_four_day = zip(*self.predict_signals_four_day)
            ax2.plot(predict_dates_four_day, predict_prices_four_day, 'o', markersize=10, color='orange', label='Four-Day Prediction Signal')
            for date, price in self.predict_signals_four_day:
                ax2.annotate(f'${price:.2f}', 
                             (date, price), 
                             xytext=(10, 0),  
                             textcoords='offset points', 
                             fontsize=9, 
                             color='orange',
                             bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', alpha=0.7))
                ax2.axvline(x=date, color='orange', linestyle='--', lw=2)
            
        ax2.set_title('Stock Price with Signals')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True)
        
        ax3.bar(TestStrategy.df.index, TestStrategy.df['volume'], color='gray', alpha=0.6, label='Volume')
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
        print(f"Mean Daily Log Return: {TestStrategy.df['log_return'][1:].mean():.4f}")
        print(f"Standard Deviation of Log Returns: {TestStrategy.df['log_return'][1:].std():.4f}")
        print(f"Cumulative Log Return: {TestStrategy.df['log_return'][1:].sum():.4f}")
        
        plt.tight_layout()
        plt.show()

def optimize_period(df):
    best_period = None
    best_performance = float('-inf')

    if len(df) < TestStrategy.LONGEST_PERIOD:
        raise ValueError("Insufficient data length for optimization")

    data_feed = bt.feeds.PandasData(dataname=df)
    periods_to_test = range(1, min(len(df) // 2, TestStrategy.LONGEST_PERIOD + 1))

    for period in periods_to_test:
        cerebro = bt.Cerebro()
        cerebro.adddata(data_feed)
        cerebro.addstrategy(TestStrategy, period=period, optimize=True)
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        result = cerebro.run()
        performance = result[0].analyzers.returns.get_analysis()['rnorm']

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

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model_one_day = LinearRegression()
    model_one_day.fit(X_poly, y_one_day)

    model_two_day = LinearRegression()
    model_two_day.fit(X_poly, y_two_day)

    model_three_day = LinearRegression()
    model_three_day.fit(X_poly, y_three_day)

    model_four_day = LinearRegression()
    model_four_day.fit(X_poly, y_four_day)

    # Attach static attributes to the strategy
    TestStrategy.df = df
    TestStrategy.poly = poly
    TestStrategy.model_one_day = model_one_day
    TestStrategy.model_two_day = model_two_day
    TestStrategy.model_three_day = model_three_day
    TestStrategy.model_four_day = model_four_day

    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(dataname=df)

    best_period = optimize_period(df)
    print(f"Best period found: {best_period}")

    # Now run with the best period, without the optimization flag
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.addstrategy(TestStrategy, period=best_period)

    result = cerebro.run()
    cerebro.plot()