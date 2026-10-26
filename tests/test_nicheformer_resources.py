"""Synthetic Nicheformer resource provenance; no authentic assets or network."""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cell_property_prediction import assets, embeddings
from cell_property_prediction.config import atomic_json, file_sha256


@pytest.fixture
def resource_fixture(tmp_path, monkeypatch):
    config = {
        "repo_dir": "vendor/Nicheformer",
        "checkpoint_dir": "checkpoint",
        "technology_mean_path": "resources/dissociated_mean.npy",
        "gene_mapping_path": "resources/mapping.tsv",
        "gene_ids_path": "resources/ids.tsv",
    }
    buffer = io.BytesIO()
    np.save(buffer, np.array([np.nan, 0., .49, .5, 1.49, 1.5, 2.5]))
    payload = buffer.getvalue()
    blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    monkeypatch.setattr(assets, "_NICHEFORMER_MEAN_BLOB", blob)
    calls = {"clone": [], "download": [], "reference": [], "mapping": [], "closed": []}
    vendor_commit = "synthetic-vendor-revision"

    def clone(url, destination, revision=None):
        calls["clone"].append((url, destination, revision))
        (destination / ".git").mkdir(parents=True)
        return {"url": url, "path": str(destination), "commit": vendor_commit}

    def download(url, destination):
        calls["download"].append((url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return {"url": url, "path": str(destination), "sha256": file_sha256(destination),
                "bytes": len(payload)}

    def read_reference(path, *, backed):
        calls["reference"].append((path, backed))
        return SimpleNamespace(n_vars=7, file=SimpleNamespace(close=lambda: calls["closed"].append(True)))

    def mapping(mapping_config, *, root):
        calls["mapping"].append((mapping_config, root))
        path = root / mapping_config["gene_mapping_path"]
        path.write_text("mouse_symbol\tmodel_gene_id\nA\tENSM1\n", encoding="utf-8")
        return {"path": str(path), "sha256": file_sha256(path)}

    monkeypatch.setattr(assets, "clone_vendor", clone)
    monkeypatch.setattr(assets, "download", download)
    monkeypatch.setattr(embeddings, "build_nicheformer_gene_mapping", mapping)
    monkeypatch.setitem(sys.modules, "anndata", SimpleNamespace(read_h5ad=read_reference))
    monkeypatch.setattr(assets.subprocess, "check_output", lambda *a, **kw: vendor_commit)
    return SimpleNamespace(root=tmp_path, config=config, payload=payload, calls=calls)


def freeze_network(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Frozen resource reuse must not download, clone, or regenerate resources")

    monkeypatch.setattr(assets, "clone_vendor", unexpected)
    monkeypatch.setattr(assets, "download", unexpected)
    monkeypatch.setattr(embeddings, "build_nicheformer_gene_mapping", unexpected)
    monkeypatch.setitem(sys.modules, "anndata", SimpleNamespace(read_h5ad=unexpected))


def test_prefetch_prepares_pinned_raw_means_with_provenance(resource_fixture):
    fixture = resource_fixture
    record = assets.prefetch_nicheformer_resources(fixture.config, root=fixture.root)
    raw = fixture.root / "resources/dissociated_mean.raw.npy"
    destination = fixture.root / fixture.config["technology_mean_path"]
    assert raw.read_bytes() == fixture.payload
    np.testing.assert_array_equal(np.load(destination), [1, 1, 1, 1, 1, 2, 3])
    assert fixture.calls["download"] == [(
        "https://raw.githubusercontent.com/theislab/nicheformer/"
        "485cadbc5caa15119adfd54228f8a8af835fcabc/data/model_means/dissociated_mean_script.npy", raw)]
    assert fixture.calls["reference"] == [(fixture.root / "checkpoint/model.h5ad", "r")]
    assert fixture.calls["closed"] == [True]
    assert len(fixture.calls["clone"]) == len(fixture.calls["mapping"]) == 1
    assert record["raw_technology_mean"]["sha256"] == file_sha256(raw)
    assert record["technology_mean"]["sha256"] == file_sha256(destination)
    assert record["normalization"]["policy"] == "dissociated_notebook_half_up_min_one"
    assert record["normalization"]["raw_nan_count"] == 1
    assert record["normalization"]["reference_genes"] == 7
    manifest = destination.parent / "prefetch.json"
    assert json.loads(manifest.read_text(encoding="utf-8")) == record
    assets.verify_resource_record(record)


def test_prefetch_reuses_frozen_resources_without_rewrites(resource_fixture, monkeypatch):
    fixture = resource_fixture
    record = assets.prefetch_nicheformer_resources(fixture.config, root=fixture.root)
    manifest = fixture.root / "resources/prefetch.json"
    saved = manifest.read_bytes()
    os.utime(manifest, (1, 1))
    frozen_time = manifest.stat().st_mtime_ns
    freeze_network(monkeypatch)
    assert assets.prefetch_nicheformer_resources(fixture.config, root=fixture.root) == record
    assert manifest.read_bytes() == saved
    assert manifest.stat().st_mtime_ns == frozen_time


@pytest.mark.parametrize("resource", ["raw_technology_mean", "technology_mean", "gene_mapping"])
def test_prefetch_rejects_changed_frozen_resources_without_download(resource_fixture, monkeypatch, resource):
    fixture = resource_fixture
    record = assets.prefetch_nicheformer_resources(fixture.config, root=fixture.root)
    manifest = fixture.root / "resources/prefetch.json"
    frozen = manifest.read_bytes()
    Path(record[resource]["path"]).write_bytes(b"changed resource")
    freeze_network(monkeypatch)
    with pytest.raises(ValueError, match="missing or changed"):
        assets.prefetch_nicheformer_resources(fixture.config, root=fixture.root)
    assert manifest.read_bytes() == frozen


def test_prefetch_rejects_legacy_manifest_without_policy(resource_fixture, monkeypatch):
    fixture = resource_fixture
    record = assets.prefetch_nicheformer_resources(fixture.config, root=fixture.root)
    record.pop("normalization")
    manifest = fixture.root / "resources/prefetch.json"
    atomic_json(manifest, record)
    frozen = manifest.read_bytes()
    freeze_network(monkeypatch)
    with pytest.raises(ValueError, match="policy"):
        assets.prefetch_nicheformer_resources(fixture.config, root=fixture.root)
    assert manifest.read_bytes() == frozen


def test_prefetch_rejects_unpinned_raw_bytes_before_preparation(resource_fixture, monkeypatch):
    fixture = resource_fixture
    monkeypatch.setattr(assets, "_NICHEFORMER_MEAN_BLOB", "0" * 40)
    with pytest.raises(ValueError, match="Git blob"):
        assets.prefetch_nicheformer_resources(fixture.config, root=fixture.root)
    assert (fixture.root / "resources/dissociated_mean.raw.npy").read_bytes() == fixture.payload
    assert not (fixture.root / "resources/dissociated_mean.npy").exists()
    assert not (fixture.root / "resources/prefetch.json").exists()
    assert fixture.calls["reference"] == fixture.calls["mapping"] == []
