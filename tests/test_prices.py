"""
тесты для src/prices.py — без реального обращения к yfinance (мокаем download),
проверяют именно то место, где ловили баги: MultiIndex-колонки и тип decision_day
при последующем merge с новостями.
"""

import pandas as pd

from src.prices import load_prices


def _fake_yfinance_response():
    """имитация ответа yfinance.download: MultiIndex-колонки (метрика, тикер)"""
    index = pd.DatetimeIndex(["2011-03-03", "2011-03-04"], name="Date")
    columns = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["NVDA"]])
    return pd.DataFrame(
        [[1.0, 2.0, 0.5, 1.5, 100], [1.1, 2.1, 0.6, 1.6, 110]],
        index=index,
        columns=columns,
    )


def test_load_prices_flattens_multiindex_columns(monkeypatch):
    monkeypatch.setattr("src.prices.yfinance.download", lambda *a, **kw: _fake_yfinance_response())

    result = load_prices("NVDA", "2011-03-03", "2011-03-05")

    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume", "decision_day"]


def test_load_prices_decision_day_is_string_like_news_side(monkeypatch):
    """
    регрессионный тест на баг: decision_day у цен должен быть строкой
    того же формата "YYYY-MM-DD", что и у новостей (assign_decision_day
    отдаёт .dt.strftime), иначе merge молча не находит совпадений.
    """
    monkeypatch.setattr("src.prices.yfinance.download", lambda *a, **kw: _fake_yfinance_response())

    result = load_prices("NVDA", "2011-03-03", "2011-03-05")

    assert result["decision_day"].tolist() == ["2011-03-03", "2011-03-04"]
    assert all(isinstance(value, str) for value in result["decision_day"])


def test_load_prices_merges_with_news_without_losing_rows(monkeypatch):
    monkeypatch.setattr("src.prices.yfinance.download", lambda *a, **kw: _fake_yfinance_response())
    prices = load_prices("NVDA", "2011-03-03", "2011-03-05")

    news = pd.DataFrame(
        {
            "decision_day": ["2011-03-03", "2011-03-04"],
            "avg_sentiment_score": [0.1, -0.2],
        }
    )

    merged = pd.merge(news, prices, on="decision_day", how="inner")

    assert len(merged) == 2
    assert merged["Open"].isna().sum() == 0
