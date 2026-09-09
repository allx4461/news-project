# содержит функцию и ее запуск
# assemble_news_with_decision_day, которая добавляет к новостям колонку decision_day

from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import AGGREGATED_OUTPUT_PATH, AGGREGATION_METHOD, DECISION_DATE_OUTPUT_PATH, END_DATE, SENTIMENT_OUTPUT_PATH, START_DATE
from src.date_processing import aggregate_news_by_decision_day, assign_decision_day, get_nasdaq_trading_days


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


if __name__ == "__main__":
    assemble_news_with_decision_day()
    aggregate_news_by_day()
