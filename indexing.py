"""Persistência transacional de chunks, embeddings e manifesto do índice ativo."""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path("data")
DEFAULT_CHROMA_DIR = DEFAULT_DATA_DIR / "chroma_db"
DEFAULT_COLLECTION_PREFIX = "cdc_consumidor"


def write_jsonl_atomic(chunks: list[dict[str, Any]], path: Path) -> bytes:
    """Grava JSONL por substituição atômica; não deixa arquivo parcialmente escrito."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in chunks).encode("utf-8")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(payload)
    os.replace(temp, path)
    return payload


def load_active_manifest(chroma_dir: Path = DEFAULT_CHROMA_DIR) -> dict[str, Any] | None:
    path = chroma_dir / "active_index.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Manifesto do índice inválido: {exc}") from exc


def prune_old_collections(
    client: Any, prefix: str, active_name: str, keep_previous: int = 1
) -> list[str]:
    """Remove coleções antigas deste projeto, mantendo a ativa e as `keep_previous` mais recentes.

    Só toca em nomes no formato gerado por `index_chunks`; falhas de remoção são ignoradas.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_[A-Za-z0-9]+_(\d{{14}})_[0-9a-f]{{6}}$")
    try:
        names = [getattr(item, "name", item) for item in client.list_collections()]
    except Exception:
        return []
    others = sorted(
        ((match.group(1), name) for name in names
         if name != active_name and (match := pattern.match(str(name)))),
        reverse=True,
    )
    deleted: list[str] = []
    for _, name in others[max(keep_previous, 0):]:
        try:
            client.delete_collection(name)
            deleted.append(name)
        except Exception:
            pass
    return deleted


def index_chunks(
    chunks: list[dict[str, Any]],
    embedding_model: Any,
    *,
    model_name: str,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_prefix: str = DEFAULT_COLLECTION_PREFIX,
    batch_size: int = 64,
    source_name: str = "documento.pdf",
    source_version: str = "não informada",
    source_url: str = "",
    source_scope: str = "automatic_cdc_range",
    extra_manifest: dict[str, Any] | None = None,
    keep_previous: int = 1,
) -> dict[str, Any]:
    """Cria uma collection versionada e troca o manifesto somente após validar tudo.

    O índice atualmente ativo não é apagado no início da ingestão. Uma falha durante a
    geração dos embeddings, a contagem ou o teste de consulta mantém a versão anterior.
    Após a troca, coleções antigas além de `keep_previous` são removidas.
    """
    if not chunks:
        raise ValueError("Não há chunks para indexar.")
    if batch_size < 1:
        raise ValueError("batch_size deve ser maior que zero.")

    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Instale chromadb para criar o índice vetorial.") from exc

    chroma_dir.mkdir(parents=True, exist_ok=True)
    document_hash = str(chunks[0].get("metadata", {}).get("document_hash", "document"))
    name_hash = re.sub(r"[^a-zA-Z0-9]", "", document_hash)[:12] or "document"
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_prefix)[:24].strip("_-") or "cdc"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    collection_name = f"{safe_prefix}_{name_hash}_{stamp}_{uuid.uuid4().hex[:6]}"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    staging = None

    try:
        staging = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": model_name},
        )
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            documents = [item["text"] for item in batch]
            passages = [f"passage: {document}" for document in documents]
            vectors = embedding_model.encode(
                passages,
                batch_size=min(batch_size, len(passages)),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            staging.add(
                ids=[item["id"] for item in batch],
                documents=documents,
                metadatas=[item["metadata"] for item in batch],
                embeddings=vectors.tolist(),
            )

        indexed_count = int(staging.count())
        if indexed_count != len(chunks):
            raise RuntimeError(
                f"Índice incompleto: {indexed_count} de {len(chunks)} chunks foram persistidos."
            )

        # Teste de consulta: garante que a collection responde a uma query real
        # antes de ela se tornar a base ativa.
        probe = embedding_model.encode(
            [f"query: {chunks[0]['text']}"],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        probe_result = staging.query(query_embeddings=[probe[0].tolist()], n_results=1)
        if not (probe_result.get("ids") or [[]])[0]:
            raise RuntimeError("Teste de consulta falhou: o índice novo não retornou resultados.")

        manifest = {
            **(extra_manifest or {}),
            "collection_name": collection_name,
            "collection_count": indexed_count,
            "embedding_model": model_name,
            "source_name": source_name,
            "source_version": source_version,
            "source_url": source_url,
            "source_scope": source_scope,
            "document_hash": document_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chroma_dir": str(chroma_dir),
        }
        manifest_path = chroma_dir / "active_index.json"
        temp_path = chroma_dir / "active_index.json.tmp"
        temp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, manifest_path)
    except Exception:
        # Apaga apenas a collection de staging desta tentativa; a ativa permanece intacta.
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        raise

    # Fora do try: uma falha na poda nunca pode apagar a collection recém-ativada.
    prune_old_collections(client, safe_prefix, collection_name, keep_previous)
    return manifest


def open_active_collection(chroma_dir: Path = DEFAULT_CHROMA_DIR):
    """Abre a collection apontada pelo manifesto ativo."""
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Instale chromadb para consultar o índice.") from exc

    manifest = load_active_manifest(chroma_dir)
    if not manifest:
        raise FileNotFoundError("Nenhum índice encontrado. Processe um PDF primeiro.")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(name=manifest["collection_name"])
    if int(collection.count()) != int(manifest.get("collection_count", -1)):
        raise RuntimeError("A contagem da collection não corresponde ao manifesto do índice.")
    return collection, manifest
