import json
import sys
import types

import numpy as np
import pytest

from indexing import index_chunks, load_active_manifest, prune_old_collections


class FakeCollection:
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata
        self.ids = []

    def add(self, ids, documents, metadatas, embeddings):
        self.ids.extend(ids)

    def count(self):
        return len(self.ids)

    def query(self, query_embeddings, n_results):
        return {"ids": [self.ids[:n_results]]}


class FakeClient:
    def __init__(self):
        self.collections = {}

    def create_collection(self, name, metadata):
        collection = FakeCollection(name, metadata)
        self.collections[name] = collection
        return collection

    def delete_collection(self, name):
        self.collections.pop(name, None)

    def list_collections(self):
        return list(self.collections)  # chroma >= 0.6 devolve nomes


class FakeModel:
    def __init__(self, fail=False):
        self.fail = fail

    def encode(self, texts, **kwargs):
        if self.fail:
            raise RuntimeError("simulated embedding failure")
        return np.ones((len(texts), 3), dtype=float)


@pytest.fixture
def fake_chroma(monkeypatch):
    client = FakeClient()
    module = types.SimpleNamespace(PersistentClient=lambda path: client)
    monkeypatch.setitem(sys.modules, "chromadb", module)
    return client


def make_chunks(document_hash="abc123def4567890"):
    return [
        {"id": "cdc-test-00001", "text": "passage one", "metadata": {"document_hash": document_hash}},
        {"id": "cdc-test-00002", "text": "passage two", "metadata": {"document_hash": document_hash}},
    ]


def test_index_is_published_only_after_all_chunks_are_written(tmp_path, fake_chroma):
    first = index_chunks(
        make_chunks(), FakeModel(), model_name="test-model", chroma_dir=tmp_path,
        source_name="cdc.pdf", source_version="2026", source_scope="automatic_cdc_range",
    )
    assert first["collection_count"] == 2
    assert first["source_scope"] == "automatic_cdc_range"
    assert load_active_manifest(tmp_path)["collection_name"] == first["collection_name"]

    previous_manifest = json.loads((tmp_path / "active_index.json").read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        index_chunks(
            make_chunks(), FakeModel(fail=True), model_name="test-model", chroma_dir=tmp_path,
            source_name="other.pdf", source_version="2027",
        )

    current_manifest = load_active_manifest(tmp_path)
    assert current_manifest == previous_manifest
    assert current_manifest["collection_name"] in fake_chroma.collections
    assert len(fake_chroma.collections) == 1


def test_manifest_keeps_full_document_hash_and_extra_fields(tmp_path, fake_chroma):
    manifest = index_chunks(
        make_chunks(), FakeModel(), model_name="m", chroma_dir=tmp_path,
        extra_manifest={"possibly_outdated": True},
    )
    assert manifest["document_hash"] == "abc123def4567890"  # 16 caracteres, igual ao JSONL
    assert manifest["possibly_outdated"] is True
    assert "abc123def456" in manifest["collection_name"]


def test_failed_smoke_test_keeps_previous_index(tmp_path, fake_chroma, monkeypatch):
    first = index_chunks(make_chunks(), FakeModel(), model_name="m", chroma_dir=tmp_path)
    monkeypatch.setattr(FakeCollection, "query", lambda self, query_embeddings, n_results: {"ids": [[]]})
    with pytest.raises(RuntimeError, match="Teste de consulta"):
        index_chunks(make_chunks(), FakeModel(), model_name="m", chroma_dir=tmp_path)
    assert load_active_manifest(tmp_path)["collection_name"] == first["collection_name"]
    assert len(fake_chroma.collections) == 1


def test_old_collections_are_pruned_but_active_and_previous_are_kept(tmp_path, fake_chroma):
    names = []
    for _ in range(4):
        names.append(index_chunks(make_chunks(), FakeModel(), model_name="m", chroma_dir=tmp_path)["collection_name"])
    assert len(fake_chroma.collections) == 2
    assert names[-1] in fake_chroma.collections


def test_prune_ignores_foreign_collections():
    client = FakeClient()
    for name in ("outra_colecao", "cdc_consumidor_abc_20200101000000_aaaaaa",
                 "cdc_consumidor_abc_20210101000000_bbbbbb", "cdc_consumidor_abc_20220101000000_cccccc"):
        client.create_collection(name, {})
    deleted = prune_old_collections(client, "cdc_consumidor", "cdc_consumidor_abc_20220101000000_cccccc", 1)
    assert deleted == ["cdc_consumidor_abc_20200101000000_aaaaaa"]
    assert "outra_colecao" in client.collections
