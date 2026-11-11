import sys
import types

import numpy as np

from retrieval import Retriever, contextualize_query, expand_query, lexical_tokens, limit_pinned, stem


class FakeBM25:
    def __init__(self, corpus):
        self.corpus = corpus

    def get_scores(self, query_tokens):
        return np.array([sum(token in doc for token in query_tokens) for doc in self.corpus])


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.array([[0.1, 0.2] for _ in texts], dtype=float)


class FakeCollection:
    def __init__(self, rows=None):
        self.rows = rows or [
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


def make_retriever(monkeypatch, collection=None):
    monkeypatch.setitem(sys.modules, "rank_bm25", types.SimpleNamespace(BM25Okapi=FakeBM25))
    return Retriever(collection or FakeCollection(), FakeEmbeddingModel())


def test_only_active_filters_lookup_vector_and_expansion(monkeypatch):
    retriever = make_retriever(monkeypatch)
    hits = retriever.search(
        "O que diz o art. 1?", top_k=2, mode="vector", lookup=True,
        only_active=True, expand=True,
    )
    assert hits
    assert all(hit["metadata"]["status"] == "vigente" for hit in hits)
    assert {hit["id"] for hit in hits}.issubset({"v1", "v2"})


def test_explicit_lookup_can_include_inactive_when_requested(monkeypatch):
    retriever = make_retriever(monkeypatch)
    hits = retriever.search(
        "O que diz o art. 1?", top_k=6, mode="bm25", lookup=True, only_active=False
    )
    assert {hit["id"] for hit in hits} >= {"v1", "r1", "m1"}


def _many_rows():
    rows = []
    for key in ("18", "19", "20"):
        for i in range(1, 9):
            rows.append((f"a{key}-{i}", f"Art. {key} parte {i}",
                         {"article_key": key, "status": "vigente", "chunk_index": int(key) * 10 + i}))
    rows.append(("sem1", "Outro assunto semântico", {"article_key": "30", "status": "vigente", "chunk_index": 999}))
    rows.append(("sem2", "Mais um assunto semântico", {"article_key": "31", "status": "vigente", "chunk_index": 1000}))
    return rows


def test_lookup_never_exceeds_top_k_and_leaves_semantic_slots(monkeypatch):
    retriever = make_retriever(monkeypatch, FakeCollection(_many_rows()))
    hits = retriever.search("arts. 18, 19 e 20", top_k=6, mode="vector", lookup=True)
    assert len(hits) <= 6
    origins = [hit["origin"] for hit in hits]
    assert origins.count("lookup") == 4  # top_k - 2 vagas semânticas
    assert any(origin != "lookup" for origin in origins)
    # rodízio: os três artigos citados aparecem, nenhum monopoliza
    assert {hit["metadata"]["article_key"] for hit in hits if hit["origin"] == "lookup"} == {"18", "19", "20"}


def test_limit_pinned_keeps_original_order():
    pinned = [{"id": f"{k}{i}", "metadata": {"article_key": k}} for k in "AB" for i in range(3)]
    kept = limit_pinned(pinned, 4)
    assert [h["id"] for h in kept] == ["A0", "A1", "B0", "B1"]
    assert limit_pinned(pinned, 10) == pinned


def test_abstention_applies_to_hybrid_without_article_reference(monkeypatch):
    retriever = make_retriever(monkeypatch)
    far = retriever.search("posso devolver o produto", top_k=3, mode="hybrid", max_distance=0.05)
    assert far == []
    near = retriever.search("posso devolver o produto", top_k=3, mode="hybrid", max_distance=0.5)
    assert near
    # artigo citado explicitamente nunca é barrado pelo limiar
    pinned = retriever.search("art. 1", top_k=3, mode="hybrid", max_distance=0.05)
    assert pinned


def test_best_distance(monkeypatch):
    retriever = make_retriever(monkeypatch)
    assert abs(retriever.best_distance("qualquer coisa") - 0.1) < 1e-9


def test_stemmer_links_lay_and_legal_forms():
    assert stem("desistir") == stem("desistencia")
    assert stem("prazos") == stem("prazo")
    assert stem("consumidor") == stem("consumidores")
    assert stem("49") == "49"


def test_bm25_tokens_ignore_chunk_labels():
    tokens = lexical_tokens("Referência: Art. 49 | TÍTULO I\n\nTrecho legal:\nO consumidor pode desistir")
    assert "referenc" not in " ".join(tokens) and "trecho" not in tokens and "legal" not in tokens
    assert stem("desistir") in tokens


def test_glossary_expands_only_lay_terms():
    assert "cadastros de inadimplentes" in expand_query("Fui negativado sem aviso")
    assert "arrependimento" in expand_query("Quero desistir da compra")
    assert expand_query("O que diz o art. 49?") == "O que diz o art. 49?"


def test_contextualize_query_uses_previous_question_for_followups():
    query = contextualize_query("E se foi pela internet?", ["Posso desistir de uma compra?"])
    assert "Posso desistir de uma compra?" in query
    assert "E se foi pela internet?" in query


def test_contextualize_query_does_not_append_old_topic_to_new_question():
    query = contextualize_query(
        "O que caracteriza uma propaganda enganosa?", ["Posso desistir de uma compra?"]
    )
    assert query == "O que caracteriza uma propaganda enganosa?"


def test_short_but_complete_question_does_not_inherit_previous_topic():
    query = contextualize_query("Qual o prazo?", ["Posso desistir de uma compra?"])
    assert query == "Qual o prazo?"
