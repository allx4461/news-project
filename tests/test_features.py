import numpy as np
import pandas as pd
import pytest

from src.features import (
    cal_anomalies,
    finbert_x_volume,
    hl_spread,
    media_attention,
    price_to_sma,
    returns,
    rolling_sentiment,
    sentiment_delta,
    volatility,
)


def test_hl_spread_is_computed_correctly():
    df = pd.DataFrame(
        {
            "High": [110.0, 90.0],
            "Low": [90.0, 90.0],
            "Close": [100.0, 90.0],
        }
    )

    hl_spread(df)

    # (110-90)/100 = 0.2 ; (90-90)/90 = 0.0
    assert df["hl_spread"].tolist() == [0.2, 0.0]


def test_media_attention_replaces_inf_with_zero_when_rolling_mean_is_zero():
    """
    регрессионный тест на деление на ноль: если новостей не было все
    последние 7 дней (rolling mean == 0), news_count / mean даёт inf,
    функция обязана заменить это на 0.0, а не протащить inf дальше.
    """
    df = pd.DataFrame({"news_count": [0] * 7})

    media_attention(df)

    assert not np.isinf(df["news_count_ratio_7d"]).any()
    assert (df["news_count_ratio_7d"] == 0.0).all()


def test_returns():
    df = pd.DataFrame({"Close": [100, 200, 400, 800, 1600, 3200]})
    returns(df)
    assert df["return_1d"].tolist()
    assert df["return_1d"].iloc[1:].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert df["return_5d"].iloc[5].tolist() == 31.0


def test_price_to_sma():
    # 20 строк подряд с одинаковой ценой, потом рост
    df = pd.DataFrame({"Close": [100.0] * 20 + [110.0]})

    price_to_sma(df)

    # первые 19 строк - NaN, sma_20 ещё не набрала окно из 20 значений
    assert df["price_to_sma_20"].iloc[:19].isna().all()
    # на 20й строке sma_20 = среднее по первым 20 значениям = 100.0 -> ratio = 1.0
    assert df["price_to_sma_20"].iloc[19] == pytest.approx(1.0)
    # на 21й строке sma_20 считается по строкам 1..20 (100*19 + 110)/20 = 100.5
    assert df["price_to_sma_20"].iloc[20] == pytest.approx(110.0 / 100.5)


def test_volatility():
    df = pd.DataFrame({"return_1d": [100, 200, 400, 800, 1600, 3200]})
    volatility(df)
    assert df["volatility_5d"].fillna(0.0).tolist() == df["return_1d"].rolling(window=5).std().fillna(0.0).tolist()


def test_finbert_x_volume():
    df = pd.DataFrame({"avg_sentiment_score": [0.5, 0.7], "volume_ratio_10d": [10, 100]})
    finbert_x_volume(df)
    assert df["finbert_x_volume"].tolist() == [5, 70]


def test_rolling_sentiment_averages_last_3_rows():
    df = pd.DataFrame({"avg_sentiment_score": [0.0, 0.2, 0.4, 0.6, 0.8]})

    rolling_sentiment(df)
    assert df["score_3d"].iloc[:2].isna().all()
    # (0.0+0.2+0.4)/3, (0.2+0.4+0.6)/3, (0.4+0.6+0.8)/3
    assert df["score_3d"].iloc[2:].tolist() == pytest.approx([0.2, 0.4, 0.6])


def test_sentiment_delta_is_zeroed_out_on_no_news_days():
    """
    регрессионный тест: в дни без новостей (has_news == 0) разница
    sentiment должна обнуляться, а не показывать "фейковое" изменение
    настроения, которое на самом деле просто дырка в данных.
    """
    df = pd.DataFrame(
        {
            "avg_sentiment_score": [0.1, 0.1, 0.9],
            "has_news": [1, 0, 1],
        }
    )

    sentiment_delta(df)

    # день 0: shift(1) -> NaN => raw_diff NaN, но has_news==1, значение не маскируется -> NaN
    assert pd.isna(df["finbert_diff_1d"].iloc[0])
    # день 1: has_news == 0 -> принудительно 0.0, даже если бы raw_diff был не 0
    assert df["finbert_diff_1d"].iloc[1] == 0.0
    # день 2: has_news == 1, raw_diff = 0.9 - 0.1 = 0.8
    assert df["finbert_diff_1d"].iloc[2] == pytest.approx(0.8)


def test_cal_anomalies_known_weekdays():
    # 2024-01-01 понедельник (dayofweek=0), 2024-01-03 среда (dayofweek=2), 2024-01-07 воскресенье (dayofweek=6)
    df = pd.DataFrame({"decision_day": ["2024-01-01", "2024-01-03", "2024-01-07"]})

    cal_anomalies(df)

    expected = [
        np.sin(2 * np.pi * 0 / 7),
        np.sin(2 * np.pi * 2 / 7),
        np.sin(2 * np.pi * 6 / 7),
    ]
    assert df["day_of_week_sin"].tolist() == pytest.approx(expected)
