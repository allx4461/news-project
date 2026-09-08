import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_finbert_model(SENTIMENT_MODEL_NAME):
    """загрузка модели FinBERT для анализа тональности новостей"""
    model_name = SENTIMENT_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    return tokenizer, model.eval()


def predict_finbert(texts: list[str], tokenizer, model, batch_size: int = 32) -> pd.DataFrame:
    """предсказание тональности новостей с помощью FinBERT,
    возвращает датафрейм-колонку с тональностью заголовка"""
    all_scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Inferencing"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",  # не нампи потому что модель на торче
            padding=True,
            truncation=True,
            max_length=128,  # больше 128 не берем для заголовков
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        pos_batch = probs[:, 0].cpu().numpy()
        neg_batch = probs[:, 1].cpu().numpy()
        score_batch = pos_batch - neg_batch
        all_scores.extend(score_batch)

    return pd.DataFrame({"sentiment_score": all_scores})
