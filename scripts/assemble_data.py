# содержит функцию и ее запуск
# assemble_news_with_decision_day, которая добавляет к новостям колонку decision_day

from pathlib import Path

import pandas as pd

from src.config import END_DATE, PROCESSED_DATA_PATH, START_DATE
from src.date_processing import assign_decision_day, get_nasdaq_trading_days


def assemble_news_with_decision_day(news_path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    добавляет к новостям колонку decision_day с ближайшей торговой датой
    для каждой даты публикации новости.
    """
    news = pd.read_csv(news_path)
    # без буфера был баг: новости после end_date откатывались в прошлое
    trading_days = get_nasdaq_trading_days(START_DATE, END_DATE + pd.Timedelta(days=90))
    news["decision_day"] = assign_decision_day(news["Date"], trading_days)
    return news


if __name__ == "__main__":
    result = assemble_news_with_decision_day()
