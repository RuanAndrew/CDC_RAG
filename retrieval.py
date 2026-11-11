"""Recuperação vetorial, BM25 e híbrida (RRF), lookup jurídico e reranking opcional."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

RRF_K = 60
MAX_LOOKUP_ARTICLES = 3
MAX_LOOKUP_CHUNKS_PER_ARTICLE = 8
MAX_LOOKUP_TOTAL = 12
# Vagas mínimas reservadas à busca semântica quando há lookup (para top_k >= 4).
MIN_SEMANTIC_SLOTS = 2

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
# Rótulos que o chunking coloca em TODO chunk; não ajudam (e poluem) o BM25.
_CHUNK_LABEL_RE = re.compile(r"^(?:Refer[êe]ncia|Trecho legal):[ \t]*", re.M)

# Sufixos removidos pelo stemmer leve (texto já sem acentos), do mais longo ao mais curto.
_SUFFIXES = tuple(sorted({
    "amentos", "imentos", "amento", "imento", "adoras", "adores", "adora", "ador",
    "acoes", "acao", "ucoes", "ucao", "encias", "encia", "ancias", "ancia",
    "anca", "ancas", "idades", "idade", "mente", "aveis", "avel", "iveis", "ivel",
    "ando", "endo", "indo", "ados", "adas", "ado", "ada", "idos", "idas", "ido",
    "ida", "ores", "or", "ar", "er", "ir", "as", "os", "es", "a", "o", "e", "s",
}, key=len, reverse=True))

# Glossário leigo → jurídico. Os padrões rodam sobre o texto sem acentos e em minúsculas.
LAY_TO_LEGAL: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"negativ|\bspc\b|serasa|nome sujo|inadimpl"),
     "cadastros de inadimplentes bancos de dados arquivos de consumo comunicação por escrito"),
    (re.compile(r"desist|arrepend|devolv|devoluc"),
     "direito de arrependimento prazo de sete dias contratação fora do estabelecimento comercial"),
    (re.compile(r"internet|online|\bsite\b|aplicativo|telefone"),
     "contratação fora do estabelecimento comercial por telefone ou a domicílio"),
    (re.compile(r"propaganda|anuncio|publicidad|oferta"),
     "publicidade enganosa abusiva oferta vincula o fornecedor"),
    (re.compile(r"defeit|quebr|estrag|nao funciona"),
     "vício do produto vício de qualidade substituição restituição abatimento do preço"),
    (re.compile(r"\bdano|acident|prejuiz"),
     "fato do produto reparação de danos responsabilidade do fornecedor"),
    (re.compile(r"cobrad|cobranc|a mais|dobro"),
     "repetição do indébito cobrança de quantia indevida valor igual ao dobro"),
    (re.compile(r"venda casada|obrig\w* a comprar|condicion"),
     "práticas abusivas condicionar fornecimento de produto ou serviço"),
    (re.compile(r"abusiv|rescis"),
     "cláusulas abusivas nulas de pleno direito contratos"),
    (re.compile(r"garantia|\btroca|trocar"),
     "garantia legal garantia contratual vício adequação"),
    (re.compile(r"superendivid|endivid|divida|juros|emprestimo|financiament"),
     "crédito financiamento informação prévia taxa de juros superendividamento"),
    (re.compile(r"quanto tempo|prazo.*reclam|reclam.*prazo"),
     "prazo decadencial para reclamar vícios aparentes ou de fácil constatação"),
)


def _plain_lower(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    plain = _plain_lower(text)
    return [t for t in re.findall(r"\w+", plain) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def stem(token: str) -> str:
    """Stemmer leve para português (texto sem acentos): 'desistir' e 'desistência' → 'desist'."""
    if token.isdigit():
        return token
    for suffix in _SUFFIXES:
        minimum = 3 if len(suffix) <= 2 else 4
        if token.endswith(suffix) and len(token) - len(suffix) >= minimum:
            return token[: -len(suffix)]
    return token


def strip_chunk_labels(text: str) -> str:
    """Remove os rótulos 'Referência:' e 'Trecho legal:' repetidos em todo chunk."""
    return _CHUNK_LABEL_RE.sub("", text)


def lexical_tokens(text: str) -> list[str]:
    """Tokens para BM25: sem rótulos de chunk, sem acentos e com stemming."""
    return [stem(token) for token in tokenize(strip_chunk_labels(text))]


def expand_query(query: str) -> str:
    """Acrescenta termos jurídicos a expressões leigas ('negativado', 'desistir'...)."""
    plain = _plain_lower(query)
    extras: list[str] = []
    for pattern, expansion in LAY_TO_LEGAL:
        if pattern.search(plain) and expansion not in extras:
            extras.append(expansion)
    if not extras:
        return query
    return f"{query}\n" + " ".join(extras)


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


def limit_pinned(pinned: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Corta o lookup em `limit` chunks sem deixar um artigo monopolizar as vagas.

    Escolhe em rodízio entre os artigos citados e devolve na ordem original.
    """
    if len(pinned) <= limit:
        return pinned
    queues: dict[str, list[dict[str, Any]]] = {}
    for hit in pinned:
        queues.setdefault(str(hit.get("metadata", {}).get("article_key", "")), []).append(hit)
    chosen: list[dict[str, Any]] = []
    groups = list(queues.values())
    while len(chosen) < limit and any(groups):
        for group in groups:
            if group and len(chosen) < limit:
                chosen.append(group.pop(0))
    position = {hit["id"]: index for index, hit in enumerate(pinned)}
    chosen.sort(key=lambda hit: position[hit["id"]])
    return chosen


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
        self._bm25 = BM25Okapi([lexical_tokens(self.records[doc_id]["text"]) for doc_id in self.ids])

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

    def best_distance(self, query: str, only_active: bool = True) -> float | None:
        """Menor distância vetorial da consulta; usada para calibrar o limiar de abstenção."""
        hits = self._vector(query.strip(), 1, only_active)
        return hits[0]["distance"] if hits else None

    def _bm25_rank(self, query: str, n: int, only_active: bool) -> list[str]:
        if not self.ids:
            return []
        query_tokens = lexical_tokens(query)
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
        keys = parse_article_refs(query)[:MAX_LOOKUP_ARTICLES]
        # Reparte o orçamento entre os artigos citados: o primeiro não pode consumir tudo.
        per_article = max(1, min(MAX_LOOKUP_CHUNKS_PER_ARTICLE, MAX_LOOKUP_TOTAL // max(len(keys), 1)))
        for key in keys:
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
            for doc_id, document, metadata in rows[:per_article]:
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
        glossary: bool = False,
    ) -> list[dict[str, Any]]:
        """Busca `top_k` trechos. O lookup de artigos citados nunca passa de `top_k`
        e deixa vagas para a busca semântica (até MIN_SEMANTIC_SLOTS).

        `max_distance` (distância de cosseno) ativa a abstenção: sem artigo citado
        explicitamente, se o melhor vizinho vetorial estiver além do limiar, devolve [].
        """
        query = query.strip()
        if not query:
            return []
        if top_k < 1 or pool < 1:
            raise ValueError("top_k e pool devem ser maiores que zero.")
        if mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError(f"Modo de recuperação inválido: {mode}")
        if rerank and self.reranker is None:
            raise ValueError("Reranking solicitado, mas nenhum modelo CrossEncoder foi carregado.")

        search_query = expand_query(query) if glossary else query
        vector_hits = self._vector(search_query, pool, only_active) if mode in {"vector", "hybrid"} else []
        bm25_ids = self._bm25_rank(search_query, pool, only_active) if mode in {"bm25", "hybrid"} else []
        pinned = self._lookup(query, only_active) if lookup else []

        if max_distance is not None and not pinned and mode in {"vector", "hybrid"}:
            if not vector_hits or vector_hits[0]["distance"] > max_distance:
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

        semantic_slots = min(MIN_SEMANTIC_SLOTS, top_k // 2)
        pinned = limit_pinned(pinned, max(1, top_k - semantic_slots))

        final: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in pinned:
            if hit["id"] in seen:
                continue
            final.append({**hit, "distance": None, "score": None, "origin": "lookup"})
            seen.add(hit["id"])

        for doc_id in order:
            if len(final) >= top_k:
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
