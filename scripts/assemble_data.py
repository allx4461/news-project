# содержит функцию и ее запуск
# assemble_news_with_decision_day, которая добавляет к новостям колонку decision_day

from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import (
    AGGREGATED_OUTPUT_PATH,
    AGGREGATED_PRICES_OUTPUT_PATH,
    AGGREGATION_METHOD,
    DECISION_DATE_OUTPUT_PATH,
    END_DATE,
    SENTIMENT_OUTPUT_PATH,
    START_DATE,
    TICKER,
)
from src.date_processing import aggregate_news_by_decision_day, assign_decision_day, get_nasdaq_trading_days
from src.prices import load_prices


def assemble_news_with_decision_day(news_path: Path = SENTIMENT_OUTPUT_PATH, output_path=DECISION_DATE_OUTPUT_PATH) -> pd.DataFrame:
    """
    добавляет к новостям колонку decision_day с ближайшей торговой датой
    для каждой даты публикации новости.
    """
    news = pd.read_csv(news_path)
    # без буфера был баг: новости после end_date откатывались в прошлое
    trading_days = get_nasdaq_trading_days(START_DATE, END_DATE + pd.Timedelta(days=90))
    news["decision_day"] = assign_decision_day(news["Date"], trading_days)
    if output_path:
        news.to_csv(output_path, index=False)
    else:
        return news


def aggregate_news_by_day(
    input_path=DECISION_DATE_OUTPUT_PATH, output_path=AGGREGATED_OUTPUT_PATH, method: Literal["mean", "median", "max", "min"] = AGGREGATION_METHOD
) -> pd.DataFrame:
    """
    агрегирует новости по колонке decision_day, считая sentiment_score по выбранному методу (mean, median, max, min)
    и количество новостей в день.
    """
    news = pd.read_csv(input_path)
    aggregated = aggregate_news_by_decision_day(news, method)
    if output_path:
        aggregated.to_csv(output_path, index=False)
    else:
        return aggregated


def add_yfinance_data(input_path=AGGREGATED_OUTPUT_PATH, output_path=AGGREGATED_PRICES_OUTPUT_PATH, ticker=TICKER) -> pd.DataFrame:
    """
    добавляет к агрегированным новостям цены акций с помощью yfinance
    """
    prices = load_prices(ticker, START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"))
    data = pd.read_csv(input_path)
    newdata = pd.merge(data, prices, on="decision_day", how="left")
    if output_path:
        newdata.to_csv(output_path, index=False)
    else:
        return newdata


if __name__ == "__main__":
    add_yfinance_data()
