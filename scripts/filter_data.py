import sys
from pathlib import Path

import pandas as pd

from src.config import CHUNKSIZE, PROCESSED_DATA_PATH, RAW_DATA_PATH, TICKER, USECOLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(PROJECT_ROOT)


def filter_ticker_streaming(
    data_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
    ticker: str = TICKER,
    chunksize: int = CHUNKSIZE,
) -> None:
    """
    чанками грузит данные по только нужному тикеру в csv-формате,
    оставляя только usecols колонки из конфига.
    также удаляет строки без даты или названия статьи, чистит лишние пробелы в строковых колонках"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    first_chunk = True
    total_rows = 0
    saved_rows = 0

    chunks = pd.read_csv(data_path, usecols=USECOLS, chunksize=chunksize)
    for chunk in chunks:
        total_rows += len(chunk)
        ticker_chunk = chunk[
            chunk["Stock_symbol"].astype("string").str.upper().eq(ticker.upper())
        ]  # они и так по формату это на случай если я решу сменить датасет (?)
        # я пока не решила будет ли критичным отсутствие текста статьи
        ticker_chunk = ticker_chunk.dropna(subset=["Date", "Article_title"])
        ticker_chunk["Article_title"] = ticker_chunk["Article_title"].astype(str).str.strip()
        # дропна на пустые строки не триггерится. доп проверка
        ticker_chunk = ticker_chunk[ticker_chunk["Article_title"] != ""]

        if not ticker_chunk.empty:
            ticker_chunk.to_csv(
                output_path,
                mode="a",  # a=add
                header=first_chunk,  # назв ст только для 1 чанка
                index=False,  # не сохраняет пандасовский индекс отдельным столбцом
            )
            first_chunk = False
            saved_rows += len(ticker_chunk)

        if total_rows % (chunksize * 100) == 0:
            print(f"overall rows: {total_rows}; saved: {saved_rows}\n")

    print(f"\n done! saved {saved_rows} rows to {output_path}\n")


if __name__ == "__main__":
    filter_ticker_streaming()
