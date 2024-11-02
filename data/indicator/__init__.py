import pandas as pd
import talib


def ema(period: int, df: pd.DataFrame):
    emas = talib.EMA(df["Close"], timeperiod=period)

    df["EMA" + str(period)] = emas


def rsi(period: int, df: pd.DataFrame):
    rsis = talib.RSI(df["Close"], timeperiod=period)

    df["RSI" + str(period)] = rsis


def macd(df: pd.DataFrame):
    df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
        df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
