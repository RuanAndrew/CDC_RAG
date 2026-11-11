"""Avalia recuperação RAG em nível de artigo, e não em nível de chunk.

Uso: python evaluate.py --questions eval/questions.csv
Configurações: vector | bm25 | hybrid, com extras +lookup +glossary +rerank
(ex.: hybrid+lookup+glossary). As métricas saem separadas por coluna `type` do CSV
(natural, lookup, updated...). Se existir eval/out_of_scope.csv, calibra também o
limiar de abstenção (--max-distance no app).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

EXTRAS = {"lookup", "rerank", "glossary"}


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
    if len(extras) != len(set(extras)) or set(extras) - EXTRAS:
        raise ValueError(f"Opções duplicadas ou desconhecidas na configuração: {name}")
    return {
        "mode": parts[0],
        "lookup": "lookup" in extras,
        "rerank": "rerank" in extras,
        "glossary": "glossary" in extras,
    }


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
                "type": (row.get("type") or "natural").strip().lower() or "natural",
            }
            for row in reader if (row.get("question") or "").strip()
        ]
    if not rows:
        raise ValueError(f"Nenhuma pergunta válida encontrada em {path}.")
    if any(not row["expected"] for row in rows):
        raise ValueError("Todas as perguntas de avaliação precisam ter expected_articles.")
    return rows


def load_question_list(path: Path) -> list[str]:
    """CSV com coluna `question` (perguntas fora de escopo, sem artigo esperado)."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "question" not in reader.fieldnames:
            raise ValueError("CSV precisa conter a coluna question.")
        return [row["question"].strip() for row in reader if (row.get("question") or "").strip()]


def calibrate_threshold(in_scope: list[float], negatives: list[float]) -> dict[str, float]:
    """Escolhe o limiar de distância (abstém se distância > limiar) que minimiza erros.

    false_refusals: perguntas válidas recusadas; false_answers: fora de escopo respondidas.
    Desempata a favor de recusar menos perguntas válidas e posiciona o limiar no meio
    entre a última distância aceita e a próxima observada.
    """
    if not in_scope or not negatives:
        raise ValueError("Informe distâncias de perguntas válidas e de fora de escopo.")
    observed = sorted(set(in_scope) | set(negatives))
    best: tuple[int, int, float] | None = None
    for threshold in observed:
        refused = sum(d > threshold for d in in_scope)
        answered = sum(d <= threshold for d in negatives)
        candidate = (refused + answered, refused, threshold)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    _, _, threshold = best
    higher = [d for d in observed if d > threshold]
    suggested = (threshold + higher[0]) / 2 if higher else threshold
    return {
        "threshold": suggested,
        "false_refusals": float(sum(d > suggested for d in in_scope)),
        "false_answers": float(sum(d <= suggested for d in negatives)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.csv"))
    parser.add_argument("--negatives", type=Path, default=Path("eval/out_of_scope.csv"))
    parser.add_argument("--chroma-dir", type=Path, default=Path("data/chroma_db"))
    parser.add_argument("--model", default=None, help="Se omitido, usa o modelo registrado no manifesto.")
    parser.add_argument("--configs", default="vector,bm25,hybrid,hybrid+glossary,hybrid+lookup,hybrid+lookup+glossary")
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
    only_active = not args.all_statuses

    rows: list[dict[str, Any]] = []
    # summary[config][type] = médias; counts[type] = nº de perguntas
    summary: dict[str, dict[str, dict[str, float]]] = {}
    counts: dict[str, int] = {}
    for question in questions:
        counts[question["type"]] = counts.get(question["type"], 0) + 1
    candidate_k = max(30, max(ks) * 5)

    for config_name, config in parsed.items():
        totals: dict[str, dict[str, float]] = {}
        for question in questions:
            hits = retriever.search(
                question["question"], top_k=candidate_k, only_active=only_active, **config
            )
            ranked = unique_preserving_order([
                str(hit.get("metadata", {}).get("article_key", "")) for hit in hits
            ])
            metrics = compute_metrics(ranked, question["expected"], ks)
            bucket = totals.setdefault(question["type"], {})
            for key, value in metrics.items():
                bucket[key] = bucket.get(key, 0.0) + value
            rows.append({
                "config": config_name,
                "type": question["type"],
                "question": question["question"],
                "expected": ";".join(sorted(question["expected"])),
                "retrieved_articles_top10": ";".join(ranked[:10]),
                **metrics,
            })
        summary[config_name] = {
            qtype: {key: value / counts[qtype] for key, value in bucket.items()}
            for qtype, bucket in totals.items()
        }

    columns = [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks] + ["mrr"]
    print(f"\nModelo {model_name} · índice {manifest['collection_name']}")
    for qtype in sorted(counts):
        print(f"\n== tipo '{qtype}' ({counts[qtype]} perguntas) ==")
        print(f"{'config':<28}" + "".join(f"{column:>11}" for column in columns))
        for name in configs:
            values = summary[name][qtype]
            print(f"{name:<28}" + "".join(f"{values[column]:>11.3f}" for column in columns))

    miss_k = 5 if 5 in ks else ks[-1]
    for name in configs:
        misses = [row for row in rows if row["config"] == name and row[f"hit@{miss_k}"] == 0.0]
        if misses:
            print(f"\n[{name}] sem acerto no top-{miss_k} artigos:")
            for row in misses:
                print(f"- ({row['type']}) {row['question']} | esperado={row['expected']} | recuperado={row['retrieved_articles_top10']}")

    if args.negatives.exists():
        negatives = load_question_list(args.negatives)
        in_scope = [q["question"] for q in questions if q["type"] != "lookup"]
        d_in = [d for q in in_scope if (d := retriever.best_distance(q, only_active)) is not None]
        d_out = [d for q in negatives if (d := retriever.best_distance(q, only_active)) is not None]
        if d_in and d_out:
            result = calibrate_threshold(d_in, d_out)
            print(
                f"\nAbstenção: distância máx. entre perguntas válidas={max(d_in):.3f}; "
                f"mín. entre fora de escopo={min(d_out):.3f}"
            )
            print(
                f"Limiar sugerido (MAX_DISTANCE): {result['threshold']:.3f} → "
                f"{int(result['false_refusals'])}/{len(d_in)} válidas recusadas, "
                f"{int(result['false_answers'])}/{len(d_out)} fora de escopo respondidas"
            )
    else:
        print(f"\n(Sem {args.negatives}: calibração de abstenção ignorada.)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResultados salvos em {args.out}")


if __name__ == "__main__":
    main()
