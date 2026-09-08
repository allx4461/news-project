from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TICKER = "NVDA"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nasdaq_exteral_data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nvda_news.csv"
START_DATE = "2011-03-03"
END_DATE = "2023-12-16"
CHUNKSIZE = 50_000
# колонки которые скачиваем
USECOLS = ["Stock_symbol", "Date", "Article_title"]

DATASET_NAME = "Zihan1004/FNSPID"
DATASET_CONFIG = "default"
SPLIT = "train"

START_DATE = pd.Timestamp("2011-03-03")
END_DATE = pd.Timestamp("2023-12-16")
