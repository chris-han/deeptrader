import yfinance as yf
import datetime
import pandas as pd


def bubble_sort(data, column):
    n = len(data)
    for i in range(n):
        for j in range(0, n - i - 1):
            if data[j][column] > data[j + 1][column]:
                data[j], data[j + 1] = data[j + 1], data[j]
    return data


def get_and_sort_hsbc_data(days=7):
    # 1. Get the data

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days)
    end_date = today

    ticker = "HSBC"
    hsbc = yf.Ticker(ticker)
    data = hsbc.history(start=start_date, end=end_date)

    if data.empty:
        print(f"No data found for {ticker} for the specified dates.")
        return

    # Convert data to a list of dictionaries for easier sorting
    data_list = data.reset_index().to_dict('records')

    # 2. Sort the data (sort by 'Close' price)
    sorted_data = bubble_sort(data_list, 'Close')

    # 3. Display the sorted data
    print(f"Sorted HSBC Data (Last {days} Days) by Ascending Closing Price:")
    for row in sorted_data:
        print(
            f"Date: {row['Date'].strftime('%Y-%m-%d')}, Open: {row['Open']:.2f}, High: {row['High']:.2f}, "
            f"Low: {row['Low']:.2f}, Close: {row['Close']:.2f}, Volume: {row['Volume']}"
        )


# Run the function
if __name__ == "__main__":
    get_and_sort_hsbc_data()