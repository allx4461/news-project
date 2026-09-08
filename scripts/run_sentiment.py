import pandas as pd

from src.config import PROCESSED_DATA_PATH, SENTIMENT_MODEL_NAME, SENTIMENT_OUTPUT_PATH
from src.sentiment import load_finbert_model, predict_finbert


def run_sentiment(news_path=PROCESSED_DATA_PATH, output_path=SENTIMENT_OUTPUT_PATH) -> None:
    """читает новости, считает sentiment_score по каждой статье, сохраняет с добавленной колонкой"""
    news = pd.read_csv(news_path)
    tokenizer, model = load_finbert_model(SENTIMENT_MODEL_NAME)
    scores = predict_finbert(news["Article_title"].tolist(), tokenizer, model)
    news["sentiment_score"] = scores["sentiment_score"]
    news.to_csv(output_path, index=False)


if __name__ == "__main__":
    run_sentiment()
