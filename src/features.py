import numpy as np
import pandas as pd


def returns(df: pd.DataFrame):
    # считаем процентное изменение за разные сроки
    df["return_1d"] = df["Close"].pct_change(1)  # day
    df["return_5d"] = df["Close"].pct_change(5)  # week
    df["return_20d"] = df["Close"].pct_change(20)  # mth


def price_to_sma(df: pd.DataFrame):
    # делим показатель на сред арифм за выбранное окно получим ratio ?> 1.0
    sma_20 = df["Close"].rolling(window=20).mean()
    df["price_to_sma_20"] = df["Close"] / sma_20


def volatility(df: pd.DataFrame):
    # стандартное отклонение дневных доходностей за выбранное окно
    df["volatility_5d"] = df["return_1d"].rolling(window=5).std()


def volume_spike(df: pd.DataFrame):
    # чем больше объем торгов тем сильнее сигнал
    df["volume_ratio_10d"] = df["Volume"] / df["Volume"].rolling(window=10).mean()


def rolling_sentiment(df: pd.DataFrame):
    # смотрим на среднее в разных окнах
    df["score_3d"] = df["avg_sentiment_score"].rolling(window=3).mean()


def sentiment_delta(df: pd.DataFrame):
    # изменение настроений за день
    raw_diff = df["avg_sentiment_score"] - df["avg_sentiment_score"].shift(1)
    df["finbert_diff_1d"] = raw_diff.where(df["has_news"] == 1, 0.0)


def media_attention(df: pd.DataFrame):
    # количество новостей относительно среднего за последнее окно
    df["news_count_ratio_7d"] = df["news_count"] / df["news_count"].rolling(window=7).mean()
    df["news_count_ratio_7d"] = df["news_count_ratio_7d"].replace([np.inf, -np.inf], 0.0).fillna(0.0)


def cal_anomalies(df: pd.DataFrame):
    # "оптимизм понедельника" и тд
    dates = pd.to_datetime(df["decision_day"])
    df["day_of_week_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)


def hl_spread(df: pd.DataFrame):
    # размах изменения в течение дня
    df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]


def finbert_x_volume(df: pd.DataFrame):
    df["finbert_x_volume"] = df["avg_sentiment_score"] * df["volume_ratio_10d"]


def count_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("decision_day").reset_index(drop=True)
    returns(df)
    price_to_sma(df)
    volatility(df)
    volume_spike(df)
    rolling_sentiment(df)
    sentiment_delta(df)
    media_attention(df)
    cal_anomalies(df)
    hl_spread(df)
    finbert_x_volume(df)
    df = df.dropna().reset_index(drop=True)
    return df
