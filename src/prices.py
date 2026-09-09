import pandas as pd
import yfinance


def load_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """загрузка цен акций с помощью yfinance"""
    data = yfinance.download(ticker, start=start_date, end=end_date)
    data.columns = data.columns.get_level_values(0)  # убрать уровень с тикером чтобы мержилось
    data = data.reset_index().rename(columns={"Date": "decision_day"})
    data["decision_day"] = pd.to_datetime(data["decision_day"]).dt.strftime("%Y-%m-%d")
    return data[["Open", "High", "Low", "Close", "Volume", "decision_day"]]
