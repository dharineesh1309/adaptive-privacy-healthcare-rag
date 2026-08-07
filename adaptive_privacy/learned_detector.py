"""Dependency-free character TF-IDF centroid prompt-injection baseline."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict

from .integrity import InjectionAssessment


class TfidfCentroidInjectionDetector:
    """A small deterministic linear baseline trained on separate labeled text.

    Documents are represented by character 3-5 gram TF-IDF vectors. Each class is
    represented by the normalized mean of its training vectors. The score is the
    cosine similarity to the injection centroid minus similarity to the benign
    centroid. A decision threshold is selected using validation data only.
    """

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 5,
        min_df: int = 2,
        max_features: int = 50_000,
    ):
        self.min_n = min_n
        self.max_n = max_n
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary: set[str] = set()
        self.idf: dict[str, float] = {}
        self.centroids: dict[int, dict[str, float]] = {0: {}, 1: {}}
        self.threshold = 0.0

    def fit(self, texts: list[str], labels: list[int]) -> None:
        if len(texts) != len(labels) or not texts:
            raise ValueError("Training texts and labels must be non-empty and aligned.")
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(self._features(text)))
        ranked = sorted(
            (
                (feature, count)
                for feature, count in document_frequency.items()
                if count >= self.min_df
            ),
            key=lambda item: (-item[1], item[0]),
        )[: self.max_features]
        self.vocabulary = {feature for feature, _ in ranked}
        total_documents = len(texts)
        self.idf = {
            feature: math.log((1 + total_documents) / (1 + frequency)) + 1.0
            for feature, frequency in ranked
        }

        sums: dict[int, dict[str, float]] = {
            0: defaultdict(float),
            1: defaultdict(float),
        }
        class_counts = Counter(int(label) for label in labels)
        if set(class_counts) != {0, 1}:
            raise ValueError("Both benign and injection training examples are required.")
        for text, raw_label in zip(texts, labels):
            label = int(raw_label)
            vector = self._vector(text)
            for feature, value in vector.items():
                sums[label][feature] += value
        for label in (0, 1):
            mean = {
                feature: value / class_counts[label]
                for feature, value in sums[label].items()
            }
            self.centroids[label] = self._normalize(mean)

    def tune_threshold(self, texts: list[str], labels: list[int]) -> float:
        scores = [self.margin(text) for text in texts]
        candidates = sorted(set([-1.0, 0.0, 1.0, *scores]))
        best: tuple[float, float, float, float] | None = None
        best_threshold = 0.0
        for threshold in candidates:
            predictions = [score >= threshold for score in scores]
            true_positive = sum(
                label == 1 and predicted
                for label, predicted in zip(labels, predictions)
            )
            false_positive = sum(
                label == 0 and predicted
                for label, predicted in zip(labels, predictions)
            )
            false_negative = sum(
                label == 1 and not predicted
                for label, predicted in zip(labels, predictions)
            )
            true_negative = len(labels) - true_positive - false_positive - false_negative
            precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            )
            recall = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            false_positive_rate = (
                false_positive / (false_positive + true_negative)
                if false_positive + true_negative
                else 0.0
            )
            accuracy = (true_positive + true_negative) / len(labels)
            objective = (f1, -false_positive_rate, accuracy, threshold)
            if best is None or objective > best:
                best = objective
                best_threshold = threshold
        self.threshold = best_threshold
        return best_threshold

    def assess(self, text: str) -> InjectionAssessment:
        margin = self.margin(text)
        detected = margin >= self.threshold
        probability_like_score = 1.0 / (1.0 + math.exp(-4.0 * margin))
        return InjectionAssessment(
            detected=detected,
            score=round(probability_like_score, 4),
            categories=["learned_injection"] if detected else [],
            matches=[],
        )

    def margin(self, text: str) -> float:
        vector = self._vector(text)
        return self._dot(vector, self.centroids[1]) - self._dot(
            vector, self.centroids[0]
        )

    def _features(self, text: str) -> list[str]:
        normalized = " " + re.sub(
            r"\s+",
            " ",
            unicodedata.normalize("NFKC", text.casefold()),
        ).strip() + " "
        return [
            normalized[index : index + size]
            for size in range(self.min_n, self.max_n + 1)
            for index in range(max(0, len(normalized) - size + 1))
        ]

    def _vector(self, text: str) -> dict[str, float]:
        counts = Counter(
            feature for feature in self._features(text) if feature in self.vocabulary
        )
        weighted = {
            feature: (1.0 + math.log(count)) * self.idf[feature]
            for feature, count in counts.items()
        }
        return self._normalize(weighted)

    @staticmethod
    def _normalize(vector: dict[str, float]) -> dict[str, float]:
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if not norm:
            return {}
        return {feature: value / norm for feature, value in vector.items()}

    @staticmethod
    def _dot(left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(feature, 0.0) for feature, value in left.items())
