"""Transparent lexical baseline and retrieval metrics."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .tokenizer import tokenize


class TfidfEncoder:
    def fit(self, documents: list[str]) -> "TfidfEncoder":
        self.vocabulary = sorted({token for text in documents for token in tokenize(text)})
        self.token_to_column = {token: index for index, token in enumerate(self.vocabulary)}
        document_frequency = Counter(token for text in documents for token in set(tokenize(text)))
        self.idf = np.array(
            [math.log((1 + len(documents)) / (1 + document_frequency[token])) + 1 for token in self.vocabulary]
        )
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(tokenize(text))
            total = max(sum(counts.values()), 1)
            for token, count in counts.items():
                if token in self.token_to_column:
                    matrix[row, self.token_to_column[token]] = count / total
        matrix *= self.idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


def retrieval_details(query_vectors: np.ndarray, document_vectors: np.ndarray) -> list[dict[str, float | int]]:
    scores = query_vectors @ document_vectors.T
    order = np.argsort(-scores, axis=1)
    ranks = np.array([int(np.where(order[index] == index)[0][0]) + 1 for index in range(len(order))])
    negative_scores = scores.copy()
    np.fill_diagonal(negative_scores, -np.inf)
    margins = np.diag(scores) - negative_scores.max(axis=1)
    return [
        {
            "rank": int(ranks[index]),
            "positive_score": float(scores[index, index]),
            "best_wrong_score": float(negative_scores[index].max()),
            "margin": float(margins[index]),
        }
        for index in range(len(ranks))
    ]


def retrieval_metrics(query_vectors: np.ndarray, document_vectors: np.ndarray) -> dict[str, float]:
    details = retrieval_details(query_vectors, document_vectors)
    ranks = np.array([row["rank"] for row in details])
    margins = np.array([row["margin"] for row in details])
    return {
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_3": float(np.mean(ranks <= 3)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_positive_negative_margin": float(margins.mean()),
    }
