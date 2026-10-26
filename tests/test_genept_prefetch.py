"""Synthetic GenePT archive layouts; no network or upstream vectors."""
import hashlib
import io
import pickle
import zipfile

import numpy as np
import pytest

from cell_property_prediction import assets, embeddings
from cell_property_prediction.config import file_sha256

NAME = "GenePT_gene_protein_embedding_model_3_text.pickle"


def stage(tmp_path, monkeypatch, members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    content = stream.getvalue()
    monkeypatch.setattr(embeddings, "GENEPT_ARCHIVE_MD5", hashlib.md5(content).hexdigest())
    config = {"vectors_path": "resources/genept_vectors.npz"}
    cache = tmp_path / "resources/GenePT_emebdding_v2.zip"
    cache.parent.mkdir()
    cache.write_bytes(content)
    monkeypatch.setattr(assets.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("cached ZIP must not download"))
    return config


@pytest.mark.parametrize("suffix", ["", "."])
def test_released_filenames_and_cached_archive(tmp_path, monkeypatch, suffix):
    vectors = {"GENEB": np.full(3072, 2, dtype=np.float32), "GENEA": np.ones(3072, dtype=np.float32)}
    member = f"GenePT_emebdding_v2/{NAME}{suffix}"
    config = stage(tmp_path, monkeypatch, {
        member: pickle.dumps(vectors),
        f"__MACOSX/folder/{NAME}{suffix}": b"metadata",
        f"__MACOSX/folder/._{NAME}{suffix}": b"metadata",
        f"folder/._{NAME}{suffix}": b"metadata",
        f"empty/{NAME}/": b"",
    })
    record = embeddings.prefetch_genept(config, root=tmp_path)
    assert record["archive_member"] == member
    assert record["genes"] == 2 and record["dimensions"] == 3072
    destination = tmp_path / config["vectors_path"]
    assert record["vectors_sha256"] == file_sha256(destination)
    symbols, values = embeddings.load_genept_vectors(destination)
    assert symbols.tolist() == ["GENEA", "GENEB"]
    np.testing.assert_array_equal(values, np.stack([vectors[symbol] for symbol in symbols]))
    assert (destination.parent / NAME).is_file()


@pytest.mark.parametrize("members", [
    [f"one/{NAME}", f"two/{NAME}"],
    [f"folder/{NAME}", f"folder/{NAME}."],
    ["folder/GenePT_gene_embedding_ada_text.pickle"],
    [f"__MACOSX/folder/{NAME}"],
    [f"folder/{NAME}.."],
    [f"folder/{NAME}/"],
])
def test_ambiguous_or_missing_vectors_rejected(tmp_path, monkeypatch, members):
    config = stage(tmp_path, monkeypatch, {name: b"not valid vectors" for name in members})
    with pytest.raises(ValueError, match="one|unique|exact"):
        embeddings.prefetch_genept(config, root=tmp_path)
    assert not (tmp_path / config["vectors_path"]).exists()


def test_checksum_before_loading_pickle(tmp_path, monkeypatch):
    config = stage(tmp_path, monkeypatch, {NAME + ".": b"not a pickle"})
    monkeypatch.setattr(embeddings, "GENEPT_ARCHIVE_MD5", "0" * 32)
    monkeypatch.setattr(embeddings, "load_genept_vectors",
                        lambda *a: pytest.fail("checksum must be checked first"))
    with pytest.raises(ValueError, match="MD5"):
        embeddings.prefetch_genept(config, root=tmp_path)
