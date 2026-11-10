"""Avalia recuperação RAG em nível de artigo, e não em nível de chunk.

Uso: python evaluate.py --questions eval/questions.csv
Configurações: vector | bm25 | hybrid | hybrid+lookup | hybrid+lookup+rerank
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_expected(raw: str) -> set[str]:
    return {part.strip().upper() for part in raw.split(";") if part.strip()}


def unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip().upper()
        if key and key not in seen:
            result.append(key)
            seen.add(key)
    return result


def compute_metrics(ranked_articles: list[str], expected: set[str], ks: list[int]) -> dict[str, float]:
    """Calcula Hit@K, Recall@K e MRR em uma lista de artigos únicos e ordenados."""
    ranked_unique = unique_preserving_order(ranked_articles)
    expected = {article.upper() for article in expected}
    metrics: dict[str, float] = {}
    for k in ks:
        found = expected.intersection(ranked_unique[:k])
        metrics[f"hit@{k}"] = float(bool(found))
        metrics[f"recall@{k}"] = len(found) / len(expected) if expected else 0.0
    rank = next(
        (position for position, article in enumerate(ranked_unique, start=1) if article in expected),
        None,
    )
    metrics["mrr"] = 1.0 / rank if rank else 0.0
    return metrics


def parse_config(name: str) -> dict[str, Any]:
    parts = name.strip().lower().split("+")
    if not parts or parts[0] not in {"vector", "bm25", "hybrid"}:
        raise ValueError(f"Configuração inválida: {name}")
    extras = parts[1:]
    if len(extras) != len(set(extras)) or set(extras) - {"lookup", "rerank"}:
        raise ValueError(f"Opções duplicadas ou desconhecidas na configuração: {name}")
    return {"mode": parts[0], "lookup": "lookup" in extras, "rerank": "rerank" in extras}


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"question", "expected_articles"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CSV precisa conter as colunas question e expected_articles.")
        rows = [
            {
                "question": row["question"].strip(),
                "expected": parse_expected(row["expected_articles"]),
                "notes": (row.get("notes") or "").strip(),
            }
            for row in reader if (row.get("question") or "").strip()
        ]
    if not rows:
        raise ValueError(f"Nenhuma pergunta válida encontrada em {path}.")
    if any(not row["expected"] for row in rows):
        raise ValueError("Todas as perguntas de avaliação precisam ter expected_articles.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.csv"))
    parser.add_argument("--chroma-dir", type=Path, default=Path("data/chroma_db"))
    parser.add_argument("--model", default=None, help="Se omitido, usa o modelo registrado no manifesto.")
    parser.add_argument("--configs", default="vector,bm25,hybrid,hybrid+lookup")
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--out", type=Path, default=Path("eval/results.csv"))
    parser.add_argument("--all-statuses", action="store_true", help="Inclui trechos revogados/vetados.")
    args = parser.parse_args()

    import chromadb
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from indexing import load_active_manifest
    from retrieval import Retriever

    ks = sorted({int(k.strip()) for k in args.ks.split(",") if k.strip()})
    if not ks or ks[0] < 1:
        parser.error("--ks precisa conter inteiros positivos, por exemplo 1,3,5,10.")
    configs = [item.strip() for item in args.configs.split(",") if item.strip()]
    parsed = {config: parse_config(config) for config in configs}
    questions = load_questions(args.questions)

    manifest = load_active_manifest(args.chroma_dir)
    if not manifest:
        raise SystemExit("Índice não encontrado. Execute a ingestão pelo app primeiro.")
    model_name = args.model or manifest["embedding_model"]
    if model_name != manifest["embedding_model"]:
        raise SystemExit(
            f"Modelo incompatível: índice={manifest['embedding_model']} e avaliação={model_name}. "
            "Use o mesmo modelo de embeddings da indexação."
        )

    collection = chromadb.PersistentClient(path=str(args.chroma_dir)).get_collection(
        manifest["collection_name"]
    )
    model = SentenceTransformer(model_name)
    reranker = CrossEncoder(args.reranker) if any(cfg["rerank"] for cfg in parsed.values()) else None
    retriever = Retriever(collection, model, reranker)

    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, float]] = {}
    candidate_k = max(30, max(ks) * 5)

    for config_name, config in parsed.items():
        totals: dict[str, float] = {}
        for question in questions:
            hits = retriever.search(
                question["question"],
                top_k=candidate_k,
                only_active=not args.all_statuses,
                **config,
            )
            ranked = unique_preserving_order([
                str(hit.get("metadata", {}).get("article_key", "")) for hit in hits
            ])
            metrics = compute_metrics(ranked, question["expected"], ks)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            rows.append({
                "config": config_name,
                "question": question["question"],
                "expected": ";".join(sorted(question["expected"])),
                "retrieved_articles_top10": ";".join(ranked[:10]),
                **metrics,
            })
        summary[config_name] = {key: value / len(questions) for key, value in totals.items()}

    columns = [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks] + ["mrr"]
    print(f"\n{len(questions)} perguntas · modelo {model_name} · índice {manifest['collection_name']}\n")
    print(f"{'config':<25}" + "".join(f"{column:>11}" for column in columns))
    for name, values in summary.items():
        print(f"{name:<25}" + "".join(f"{values[column]:>11.3f}" for column in columns))

    miss_k = 5 if 5 in ks else ks[-1]
    for name in configs:
        misses = [row for row in rows if row["config"] == name and row[f"hit@{miss_k}"] == 0.0]
        if misses:
            print(f"\n[{name}] sem acerto no top-{miss_k} artigos:")
            for row in misses:
                print(f"- {row['question']} | esperado={row['expected']} | recuperado={row['retrieved_articles_top10']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResultados salvos em {args.out}")


if __name__ == "__main__":
    main()
