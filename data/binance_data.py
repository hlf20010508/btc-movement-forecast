import pandas as pd
from binance.client import Client
import indicator

client = Client()


def get_market_data(symbol, interval, start_str, end_str=None):
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)

    if len(klines) > 0:
        data = pd.DataFrame(klines).iloc[:, :6]

        data.columns = ["date", "Open", "High", "Low", "Close", "Volume"]

        data["date"] = pd.to_datetime(data["date"], unit="ms")
        # data = data.set_index("date")

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            data[col] = pd.to_numeric(data[col])

        return data


def add_indicator(data: pd.DataFrame):
    indicator.ema(7, data)
    indicator.ema(14, data)
    indicator.ema(21, data)

    indicator.rsi(7, data)
    indicator.rsi(14, data)
    indicator.rsi(21, data)

    indicator.macd(data)


print("Downloading BTCUSDT data...")

btc_data = get_market_data(
    "BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "4 YEAR UTC", "10 DAY UTC"
)
# btc_data = pd.read_csv("data/btcusdt.csv")
add_indicator(btc_data)
btc_data.dropna(inplace=True)
btc_data.to_csv("data/btcusdt_train.csv", index=False)
print("Saved data to data/btcusdt_train.csv")

btc_data = get_market_data("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "10 DAY UTC")
btc_data.to_csv("data/btcusdt_valid.csv")
print("Saved data to data/btcusdt_valid.csv")
