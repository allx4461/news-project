import pandas as pd

from src.date_processing import aggregate_news_by_decision_day, assign_decision_day, get_nasdaq_trading_days


def test_semantic_1_day():
    news = pd.DataFrame(
        {
            "Date": ["2023-12-14 15:00:00 UTC", "2023-12-14 22:00:00 UTC"],
            "Article_title": ["news1", "news2"],
            "sentiment_score": [0.5, -0.2],
        }
    )
    trading_days = get_nasdaq_trading_days("2023-12-01", "2023-12-20")
    news["decision_day"] = assign_decision_day(news["Date"], trading_days)
    aggregated = aggregate_news_by_decision_day(news, method="mean")
    assert aggregated.shape[0] == 2  # две даты публикации -> два дня
    assert aggregated.loc[aggregated["decision_day"] == "2023-12-14", "avg_sentiment_score"].iloc[0] == 0.5
    assert aggregated.loc[aggregated["decision_day"] == "2023-12-15", "avg_sentiment_score"].iloc[0] == -0.2


def test_2_news_same_day():
    news = pd.DataFrame(
        {
            "Date": ["2023-12-14 11:00:00 UTC", "2023-12-14 15:00:00 UTC"],
            "Article_title": ["news1", "news2"],
            "sentiment_score": [0.5, -0.2],
        }
    )
    trading_days = get_nasdaq_trading_days("2023-12-01", "2023-12-20")
    news["decision_day"] = assign_decision_day(news["Date"], trading_days)
    aggregated = aggregate_news_by_decision_day(news, method="mean")
    assert aggregated.shape[0] == 1  # две даты публикации -> один день
    assert (
        aggregated.loc[
            aggregated["decision_day"] == "2023-12-14",
            # среднее двух новостей
            "avg_sentiment_score",
        ].iloc[0]
        == 0.15
    )


def test_nan_sentiment():
    news = pd.DataFrame(
        {
            "Date": ["2023-12-14 11:00:00 UTC", "2023-12-14 15:00:00 UTC"],
            "Article_title": ["news1", "news2"],
            "sentiment_score": [0.5, None],
        }
    )
    trading_days = get_nasdaq_trading_days("2023-12-01", "2023-12-20")
    news["decision_day"] = assign_decision_day(news["Date"], trading_days)
    aggregated = aggregate_news_by_decision_day(news, method="mean")
    assert aggregated.shape[0] == 1  # две даты публикации -> один день
    # среднее двух новостей, одна из которых NaN
    assert aggregated.loc[aggregated["decision_day"] == "2023-12-14", "avg_sentiment_score"].iloc[0] == 0.5
