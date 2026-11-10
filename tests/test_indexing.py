import json
import sys
import types

import numpy as np
import pytest

from indexing import index_chunks, load_active_manifest


class FakeCollection:
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata
        self.ids = []

    def add(self, ids, documents, metadatas, embeddings):
        self.ids.extend(ids)

    def count(self):
        return len(self.ids)


class FakeClient:
    def __init__(self):
        self.collections = {}

    def create_collection(self, name, metadata):
        collection = FakeCollection(name, metadata)
        self.collections[name] = collection
        return collection

    def delete_collection(self, name):
        self.collections.pop(name, None)


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


def make_chunks():
    return [
        {"id": "cdc-test-00001", "text": "passage one", "metadata": {"document_hash": "abc123"}},
        {"id": "cdc-test-00002", "text": "passage two", "metadata": {"document_hash": "abc123"}},
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
