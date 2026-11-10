"""Recuperação vetorial, BM25 e híbrida (RRF), lookup jurídico e reranking opcional."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

RRF_K = 60
MAX_LOOKUP_ARTICLES = 3
MAX_LOOKUP_CHUNKS_PER_ARTICLE = 8
MAX_LOOKUP_TOTAL = 12

ARTICLE_REF_RE = re.compile(
    r"\b(?:artigos?|arts?)\.?\s*((?:\d{1,3}(?:-[A-Za-z])?"
    r"(?:\s*(?:,|e)\s*(?=\d))?)+)", re.I
)
QUESTION_START_RE = re.compile(
    r"^\s*(?:o\s+que|qual|quais|quem|quando|quanto|quantos|quantas|como|onde|por\s+que|art(?:igo)?\.?)\b",
    re.I,
)
FOLLOWUP_RE = re.compile(
    r"^\s*(?:e\s+(?:se|nesse|nessa|no|na|para|com|quando|caso)|"
    r"(?:nesse|nessa)\s+caso|isso|tamb[eé]m|e\s+(?:o|a|os|as)\b)",
    re.I,
)
STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "no", "na",
    "nos", "nas", "um", "uma", "que", "se", "por", "para", "com", "ao", "aos",
    "ou", "eu", "posso", "pode", "me", "sobre", "qual", "quais", "como", "quando",
}


def tokenize(text: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(ch for ch in plain if not unicodedata.combining(ch))
    return [t for t in re.findall(r"\w+", plain) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def contextualize_query(query: str, previous_user_questions: list[str] | None = None) -> str:
    """Acrescenta a última pergunta anterior somente quando a atual parece um follow-up.

    É uma heurística barata para recuperar o assunto implícito em perguntas como
    "e se foi pela internet?" sem misturar o histórico inteiro em toda consulta.
    """
    current = query.strip()
    history = [item.strip() for item in (previous_user_questions or []) if item.strip()]
    if not current or not history:
        return current
    is_short_followup = len(current.split()) <= 4 and not QUESTION_START_RE.match(current)
    if is_short_followup or FOLLOWUP_RE.match(current):
        return f"{history[-1]}\nPergunta de acompanhamento: {current}"
    return current


def parse_article_refs(query: str) -> list[str]:
    """Extrai referências explícitas: 'art. 49', 'arts. 18 e 19', 'art. 54-A'."""
    keys: list[str] = []
    for group in ARTICLE_REF_RE.findall(query):
        for num, suffix in re.findall(r"(\d{1,3})(?:-([A-Za-z]))?", group):
            key = f"{num}-{suffix.upper()}" if suffix else num
            if key not in keys:
                keys.append(key)
    return keys


def rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion para combinar listas sem normalização de score."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
    return scores


def _where(*conditions: dict[str, Any]) -> dict[str, Any] | None:
    valid = tuple(condition for condition in conditions if condition)
    if not valid:
        return None
    return valid[0] if len(valid) == 1 else {"$and": list(valid)}


def _active(metadata: dict[str, Any]) -> bool:
    # Fail closed: metadados ausentes/desconhecidos não são tratados como vigentes.
    return metadata.get("status") == "vigente"


class Retriever:
    """Recuperador sobre uma collection Chroma já indexada com prefixo E5 `passage:`."""

    def __init__(self, collection: Any, model: Any, reranker: Any | None = None) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("Instale rank-bm25 para habilitar a recuperação lexical.") from exc

        self.collection = collection
        self.model = model
        self.reranker = reranker
        data = collection.get(include=["documents", "metadatas"])
        self.ids: list[str] = list(data.get("ids") or [])
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        self.records: dict[str, dict[str, Any]] = {
            doc_id: {"id": doc_id, "text": document or "", "metadata": metadata or {}}
            for doc_id, document, metadata in zip(self.ids, documents, metadatas)
        }
        self._bm25 = BM25Okapi([tokenize(self.records[doc_id]["text"]) for doc_id in self.ids])

    def _vector(self, query: str, n: int, only_active: bool) -> list[dict[str, Any]]:
        count = int(self.collection.count())
        if count == 0:
            return []
        embedding = self.model.encode(
            [f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        result = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(n, count),
            where=_where({"status": "vigente"} if only_active else {}),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            {"id": doc_id, "text": doc, "metadata": meta or {}, "distance": float(distance)}
            for doc_id, doc, meta, distance in zip(ids, docs, metas, distances)
            if not only_active or _active(meta or {})
        ]

    def _bm25_rank(self, query: str, n: int, only_active: bool) -> list[str]:
        if not self.ids:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.ids, scores), key=lambda item: (-float(item[1]), item[0]))
        output: list[str] = []
        for doc_id, score in ranked:
            if float(score) <= 0:
                break
            if only_active and not _active(self.records[doc_id]["metadata"]):
                continue
            output.append(doc_id)
            if len(output) >= n:
                break
        return output

    def _lookup(self, query: str, only_active: bool) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        if not hasattr(self.collection, "get"):
            return hits
        for key in parse_article_refs(query)[:MAX_LOOKUP_ARTICLES]:
            conditions: list[dict[str, Any]] = [{"article_key": key}]
            if only_active:
                conditions.append({"status": "vigente"})
            result = self.collection.get(
                where=_where(*conditions), include=["documents", "metadatas"]
            )
            rows = list(zip(
                result.get("ids", []), result.get("documents", []), result.get("metadatas", [])
            ))
            rows.sort(key=lambda row: int((row[2] or {}).get("chunk_index", 0)))
            for doc_id, document, metadata in rows[:MAX_LOOKUP_CHUNKS_PER_ARTICLE]:
                if only_active and not _active(metadata or {}):
                    continue
                hits.append({"id": doc_id, "text": document, "metadata": metadata or {}})
        return hits[:MAX_LOOKUP_TOTAL]

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        pool: int = 30,
        lookup: bool = True,
        only_active: bool = True,
        rerank: bool = False,
        expand: bool = False,
        max_distance: float | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        if top_k < 1 or pool < 1:
            raise ValueError("top_k e pool devem ser maiores que zero.")
        if mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError(f"Modo de recuperação inválido: {mode}")
        if rerank and self.reranker is None:
            raise ValueError("Reranking solicitado, mas nenhum modelo CrossEncoder foi carregado.")

        vector_hits = self._vector(query, pool, only_active) if mode in {"vector", "hybrid"} else []
        bm25_ids = self._bm25_rank(query, pool, only_active) if mode in {"bm25", "hybrid"} else []
        pinned = self._lookup(query, only_active) if lookup else []

        if max_distance is not None and vector_hits and not pinned:
            if (mode == "vector" or not bm25_ids) and vector_hits[0]["distance"] > max_distance:
                return []

        candidates: dict[str, dict[str, Any]] = {}
        for hit in vector_hits:
            candidates[hit["id"]] = {**hit, "origin": "vector"}
        for doc_id in bm25_ids:
            if doc_id not in candidates:
                candidates[doc_id] = {**self.records[doc_id], "distance": None, "origin": "bm25"}

        rankings: list[list[str]] = []
        if vector_hits:
            rankings.append([hit["id"] for hit in vector_hits])
        if bm25_ids:
            rankings.append(bm25_ids)
        scores = rrf(rankings)
        order = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        for doc_id in order:
            candidates[doc_id]["score"] = scores[doc_id]

        if rerank and self.reranker is not None and order:
            head = order[:pool]
            ce_scores = self.reranker.predict([(query, candidates[doc_id]["text"]) for doc_id in head])
            for doc_id, score in zip(head, ce_scores):
                candidates[doc_id]["score"] = float(score)
            order = sorted(head, key=lambda doc_id: (-candidates[doc_id]["score"], doc_id))

        final: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in pinned:
            if hit["id"] in seen:
                continue
            final.append({**hit, "distance": None, "score": None, "origin": "lookup"})
            seen.add(hit["id"])

        for doc_id in order:
            if len(final) >= max(top_k, min(len(pinned), top_k + 2)):
                break
            if doc_id in seen:
                continue
            final.append(candidates[doc_id])
            seen.add(doc_id)

        if expand:
            keys = list(dict.fromkeys(
                hit.get("metadata", {}).get("article_key") for hit in final[:2]
                if hit.get("metadata", {}).get("article_key")
            ))
            for key in keys:
                siblings = [
                    record for record in self.records.values()
                    if record["metadata"].get("article_key") == key
                    and record["id"] not in seen
                    and (not only_active or _active(record["metadata"]))
                ]
                siblings.sort(key=lambda r: int(r["metadata"].get("chunk_index", 0)))
                for sibling in siblings[:3]:
                    final.append({**sibling, "distance": None, "score": None, "origin": "expand"})
                    seen.add(sibling["id"])
        return final
