"""Small dependency-free BM25 retriever for the lab demonstration."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .models import DocumentVariants, RetrievalResult


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "give",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "she",
    "the",
    "their",
    "to",
    "was",
    "were",
    "what",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if term not in STOP_WORDS
    ]


class BM25Retriever:
    def __init__(
        self,
        documents: list[DocumentVariants],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not documents:
            raise ValueError("At least one tiered document is required.")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter[str]] = []
        self.document_lengths: list[int] = []
        document_frequency: dict[str, int] = defaultdict(int)

        for document in documents:
            terms = tokenize(document.low_text)
            frequencies = Counter(terms)
            self.term_frequencies.append(frequencies)
            self.document_lengths.append(len(terms))
            for term in frequencies:
                document_frequency[term] += 1

        self.average_length = sum(self.document_lengths) / len(self.document_lengths)
        total = len(documents)
        self.idf = {
            term: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        query_terms = tokenize(query)
        scored: list[tuple[int, float]] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length = self.document_lengths[index]
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / max(self.average_length, 1.0)
                )
                score += self.idf[term] * frequency * (self.k1 + 1.0) / denominator
            if score > 0:
                scored.append((index, score))

        scored.sort(key=lambda item: (-item[1], self.documents[item[0]].document_id))
        return [
            RetrievalResult(
                document_id=self.documents[index].document_id,
                score=round(score, 4),
                document=self.documents[index],
            )
            for index, score in scored[: max(1, top_k)]
        ]
