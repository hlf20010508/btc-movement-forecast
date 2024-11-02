import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from data.indicator import add_indicator

try:
    client = Client()
except BinanceAPIException:
    # volume will be very low
    client = Client(tld="us")


def get_market_data(symbol, interval, start_str, end_str=None):
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)

    if len(klines) > 0:
        data = pd.DataFrame(klines).iloc[:, :6]

        data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

        data["Date"] = pd.to_datetime(data["Date"], unit="ms")

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            data[col] = pd.to_numeric(data[col])

        return data


def fetch():
    print("Downloading BTCUSDT data...")

    btc_data = get_market_data(
        "BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "4 YEAR UTC", "10 DAY UTC"
    )

    add_indicator(btc_data)
    btc_data.dropna(inplace=True)
    btc_data.to_csv("data/btcusdt_train.csv", index=False)
    print("Saved data to data/btcusdt_train.csv")

    btc_data = get_market_data("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "10 DAY UTC")
    btc_data.to_csv("data/btcusdt_valid.csv", index=False)
    print("Saved data to data/btcusdt_valid.csv")
