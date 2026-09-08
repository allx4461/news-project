from pathlib import Path

import pandas as pd

from src.config import CHUNKSIZE, RAW_DATA_PATH

DATA_PATH = RAW_DATA_PATH


def explore_data(data_path: Path = DATA_PATH, chunksize: int = CHUNKSIZE) -> None:
    """вывод информации о данных по принимаемому пути (всего рядов и колонок,
    пропущенных значений и разброс дат)"""
    total_rows = 0
    missing_values = {}
    date_min = None
    date_max = None

    for chunk_number, chunk in enumerate(pd.read_csv(data_path, chunksize=chunksize), start=1):
        total_rows += len(chunk)

        for column, count in chunk.isna().sum().items():
            missing_values[column] = missing_values.get(column, 0) + count

        dates = pd.to_datetime(chunk["Date"], errors="coerce")
        chunk_min = dates.min()
        chunk_max = dates.max()
        date_min = chunk_min if date_min is None else min(date_min, chunk_min)
        date_max = chunk_max if date_max is None else max(date_max, chunk_max)

        if chunk_number % 10 == 0:
            print(f"Processed rows: {total_rows}")

    print(f"rows: {total_rows}")
    print(f"cols: {list(chunk.columns)}")
    print(f"dates: {date_min} to {date_max}")
    print("missing:")
    print(missing_values)


if __name__ == "__main__":
    explore_data()
