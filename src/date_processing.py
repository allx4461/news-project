# этот файл содержит функции:
# получение торговых дней NASDAQ в нужный период
# определение ближайшей торговой даты для каждой даты публикации новости
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


def get_nasdaq_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    получение торговых дней NASDAQ в нужный период.
    """
    nasdaq = mcal.get_calendar("NASDAQ")
    schedule = nasdaq.schedule(start_date=start, end_date=end)
    return schedule.index.normalize()


def assign_decision_day(
    dates: pd.Series, trading_days: pd.DatetimeIndex, market_close_hour: int = 16
) -> pd.Series:
    """
    определение ближайшей торговой даты для каждой даты публикации новости.
    (если новость опубликована после закрытия рынка, то ближайшей торговой датой считается
    следующий торговый день)
    """
    dates = pd.to_datetime(dates).dt.tz_convert("America/New_York")
    day = dates.dt.normalize()
    after_close = dates.dt.hour >= market_close_hour
    decision_day = day.where(~after_close, day + pd.Timedelta(days=1))
    decision_day = decision_day.dt.tz_localize(None)  # trading_days из mcal тоже tz-naive

    td_values = trading_days.values
    idx = np.searchsorted(td_values, decision_day.values, side="left")
    idx = np.clip(idx, 0, len(td_values) - 1)
    return pd.Series(td_values[idx], index=dates.index).dt.strftime("%Y-%m-%d")
