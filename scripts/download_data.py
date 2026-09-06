
from datasets import load_dataset
import pandas as pd
from pathlib import Path
from itertools import islice

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import CHUNKSIZE, RAW_DATA_PATH, USECOLS, DATASET_NAME, DATASET_CONFIG, SPLIT

def iter_chunks(stream, chunk_size: int):
    """итератор чанков"""
    iterator = iter(stream)
    while chunk := list(islice(iterator, chunk_size)):
        yield chunk


def download_dataset(
    output_path: Path = RAW_DATA_PATH,
    chunk_size: int = CHUNKSIZE,
    cols: list[str] = USECOLS
) -> None:
    """лениво скачивает датасет нужные колонки и чанками укладывает его в хард """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    stream = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=SPLIT,
        streaming=True,
    )

    first_chunk = True
    total_rows = 0

    for rows in iter_chunks(stream, chunk_size):
        chunk = pd.DataFrame(rows, columns=cols)
        chunk.to_csv(
            output_path,
            mode="a",  # дописываем
            header=first_chunk,  # нет дубликатов названий столбцов
            index=False,  # отказ от пандасовских индексов
        )
        first_chunk = False
        total_rows += len(chunk)
        print(f"downloaded {total_rows} rows\n", flush=True)

    print(f"done! {total_rows} rows saved to {output_path}")


if __name__ == "__main__":
    download_dataset()
