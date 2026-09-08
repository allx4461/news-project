"""
тесты для src/date_processing.py — проверяют случаи с утечкой
на прошлую торговую дату, с таймзоной и с выходными днями
"""

import pandas as pd

from src.date_processing import assign_decision_day, get_nasdaq_trading_days


def test_get_nasdaq_trading_days_excludes_weekends():
    trading_days = get_nasdaq_trading_days("2023-12-11", "2023-12-17")
    # 16 и 17 декабря 2023 — суббота и воскресенье - неторговые дни
    assert pd.Timestamp("2023-12-16") not in trading_days
    assert pd.Timestamp("2023-12-17") not in trading_days
    assert pd.Timestamp("2023-12-15") in trading_days  # пятница — есть


def test_news_before_close_same_day():
    trading_days = get_nasdaq_trading_days("2023-12-01", "2023-12-20")
    dates = pd.Series(["2023-12-14 15:00:00 UTC"])  # 10:00 NY, будний день
    result = assign_decision_day(dates, trading_days)
    assert result.iloc[0] == "2023-12-14"


def test_news_after_close_next_trading_day():
    trading_days = get_nasdaq_trading_days("2023-12-01", "2023-12-20")
    dates = pd.Series(["2023-12-14 22:00:00 UTC"])  # 17:00 NY, после закрытия
    result = assign_decision_day(dates, trading_days)
    assert result.iloc[0] == "2023-12-15"


def test_weekend_news_rounds_forward_not_backward():
    """
    тест против утечки: новость в выходной должна уйти на следующий
    торговый день, а не откатиться на прошлый (что было в прошлой версии.)
    """
    trading_days = get_nasdaq_trading_days("2023-12-01", "2023-12-20")
    dates = pd.Series(["2023-12-16 15:00:00 UTC"])  # суббота
    result = assign_decision_day(dates, trading_days)
    assert result.iloc[0] == "2023-12-18"  # понедельник, а не пятница 15.12


def test_hour_is_compared_in_new_york_time_not_utc():
    """
    регрессионный тест на баг с таймзоной: 20:00 UTC зимой = 15:00 NY,
    то есть ДО закрытия рынка (16:00 NY) -> тот же торговый день.
    """
    trading_days = get_nasdaq_trading_days("2023-12-01", "2023-12-20")
    dates = pd.Series(["2023-12-14 20:00:00 UTC"])  # 15:00 NY, будний день
    result = assign_decision_day(dates, trading_days)
    assert result.iloc[0] == "2023-12-14"
