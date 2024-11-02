from okx.app import OkxSPOT
import pandas as pd
from datetime import datetime, timedelta
from indicator import add_indicator


market = OkxSPOT(key="", secret="", passphrase="").market


def get_market_data(symbol, interval, start, end=datetime.now()):
    klines = market.get_history_candle(
        instId=symbol,
        start=start,
        end=end,
        bar=interval,
    )["data"]

    if len(klines) > 0:
        data = pd.DataFrame(klines).iloc[:, :6]

        data.columns = ["date", "Open", "High", "Low", "Close", "Volume"]

        data["date"] = pd.to_datetime(data["date"], unit="ms")

        return data


now = datetime.now()

print("Downloading BTCUSDT data...")

btc_data = get_market_data(
    "BTC-USDT", "1H", now - timedelta(days=365 * 4), now - timedelta(days=10)
)
# btc_data = pd.read_csv("data/btcusdt.csv")
add_indicator(btc_data)
btc_data.dropna(inplace=True)
btc_data.to_csv("data/btcusdt_train.csv", index=False)
print("Saved data to data/btcusdt_train.csv")

btc_data = get_market_data("BTC-USDT", "1H", now - timedelta(days=10))
btc_data.to_csv("data/btcusdt_valid.csv")
print("Saved data to data/btcusdt_valid.csv")
