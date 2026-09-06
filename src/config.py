from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TICKER = "NVDA"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nasdaq_exteral_data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nvda_news.csv"
CHUNKSIZE = 50_000
USECOLS = ["Stock_symbol", "Date", "Article_title"]#колонки которые скачиваем

DATASET_NAME = "Zihan1004/FNSPID"
DATASET_CONFIG = "default"
SPLIT = "train"