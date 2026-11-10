import sys
import types

import numpy as np

from retrieval import Retriever


class FakeBM25:
    def __init__(self, corpus):
        self.corpus = corpus

    def get_scores(self, query_tokens):
        return np.array([sum(token in doc for token in query_tokens) for doc in self.corpus])


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.array([[0.1, 0.2] for _ in texts], dtype=float)


class FakeCollection:
    def __init__(self):
        self.rows = [
            ("v1", "Art. 1 caput e regra geral", {"article_key": "1", "status": "vigente", "chunk_index": 1}),
            ("r1", "Art. 1 parágrafo revogado", {"article_key": "1", "status": "revogado", "chunk_index": 2}),
            ("m1", "Art. 1 trecho misto", {"article_key": "1", "status": "misto", "chunk_index": 3}),
            ("v2", "Art. 1 outro dispositivo vigente", {"article_key": "1", "status": "vigente", "chunk_index": 4}),
        ]

    def count(self):
        return len(self.rows)

    def get(self, where=None, include=None):
        rows = [row for row in self.rows if self._matches(row[2], where)]
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [r[2] for r in rows],
        }

    def query(self, query_embeddings, n_results, where=None, include=None):
        rows = [row for row in self.rows if self._matches(row[2], where)][:n_results]
        return {
            "ids": [[r[0] for r in rows]],
            "documents": [[r[1] for r in rows]],
            "metadatas": [[r[2] for r in rows]],
            "distances": [[0.1 + i * 0.01 for i, _ in enumerate(rows)]],
        }

    @classmethod
    def _matches(cls, metadata, where):
        if not where:
            return True
        if "$and" in where:
            return all(cls._matches(metadata, condition) for condition in where["$and"])
        return all(metadata.get(key) == value for key, value in where.items())


def test_only_active_filters_lookup_vector_and_expansion(monkeypatch):
    monkeypatch.setitem(sys.modules, "rank_bm25", types.SimpleNamespace(BM25Okapi=FakeBM25))
    retriever = Retriever(FakeCollection(), FakeEmbeddingModel())

    hits = retriever.search(
        "O que diz o art. 1?", top_k=2, mode="vector", lookup=True,
        only_active=True, expand=True,
    )
    assert hits
    assert all(hit["metadata"]["status"] == "vigente" for hit in hits)
    assert {hit["id"] for hit in hits}.issubset({"v1", "v2"})


def test_explicit_lookup_can_include_inactive_when_requested(monkeypatch):
    monkeypatch.setitem(sys.modules, "rank_bm25", types.SimpleNamespace(BM25Okapi=FakeBM25))
    retriever = Retriever(FakeCollection(), FakeEmbeddingModel())

    hits = retriever.search(
        "O que diz o art. 1?", top_k=2, mode="bm25", lookup=True, only_active=False
    )
    assert {hit["id"] for hit in hits} >= {"v1", "r1", "m1"}


def test_contextualize_query_uses_previous_question_for_followups():
    from retrieval import contextualize_query

    query = contextualize_query(
        "E se foi pela internet?", ["Posso desistir de uma compra?"]
    )
    assert "Posso desistir de uma compra?" in query
    assert "E se foi pela internet?" in query


def test_contextualize_query_does_not_append_old_topic_to_new_question():
    from retrieval import contextualize_query

    query = contextualize_query(
        "O que caracteriza uma propaganda enganosa?", ["Posso desistir de uma compra?"]
    )
    assert query == "O que caracteriza uma propaganda enganosa?"


def test_short_but_complete_question_does_not_inherit_previous_topic():
    from retrieval import contextualize_query

    query = contextualize_query("Qual o prazo?", ["Posso desistir de uma compra?"])
    assert query == "Qual o prazo?"
