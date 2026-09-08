"""
тесты для src/sentiment.py на фейковых tokenizer/model (без скачивания весов) —
проверяют только логику индексов/батчинга, а не саму модель.
"""

import torch

from src.sentiment import predict_finbert


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return FakeBatch(n=len(texts))


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    def __call__(self, n):
        # id2label ProsusAI/finbert: 0=positive, 1=negative, 2=neutral
        return FakeOutput(torch.tensor([[10.0, 0.0, 0.0]] * n))


def test_predict_finbert_positive_label_is_index_0():
    result = predict_finbert(["great news"], FakeTokenizer(), FakeModel(), batch_size=1)
    assert result["sentiment_score"].iloc[0] > 0.9


def test_predict_finbert_processes_all_texts_across_batches():
    texts = ["a", "b", "c"]
    result = predict_finbert(texts, FakeTokenizer(), FakeModel(), batch_size=2)
    assert len(result) == 3


def test_predict_fibert_no_input():
    result = predict_finbert([], FakeTokenizer(), FakeModel(), batch_size=2)
    assert len(result) == 0


def test_predict_finbert_batch_size_larger_than_texts():
    texts = ["a", "b"]
    result = predict_finbert(texts, FakeTokenizer(), FakeModel(), batch_size=10)
    assert len(result) == 2
