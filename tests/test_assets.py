"""Synthetic asset contracts; network and authentic checkpoints are never used."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cell_property_prediction import assets
from cell_property_prediction.config import atomic_json, file_sha256


def fixture_config():
    return yaml.safe_load((Path(__file__).parents[1] / "configs" / "benchmark.yaml").read_text())


def test_prefetch_pins_kumo_package_checkpoint_and_reuses_frozen_manifest(tmp_path, monkeypatch):
    config = fixture_config()
    cache = tmp_path / config["paths"]["cache_dir"]
    cache.mkdir()
    original_tabpfn = {"untouched_fixture": True}
    atomic_json(cache / "model_manifest.json", {"tabpfn": original_tabpfn})
    calls = []

    def snapshot(repo_id, destination, hub_cache, revision):
        calls.append((repo_id, revision))
        destination.mkdir(parents=True)
        (destination / "large").mkdir()
        checkpoint = destination / "large" / "regressor.pt"
        checkpoint.write_bytes(b"synthetic-checkpoint")
        return {"repo_id": repo_id, "revision": "immutable-fixture", "path": str(destination),
                "files": [{"path": "large/regressor.pt", "sha256": file_sha256(checkpoint)}]}

    monkeypatch.setattr(assets, "hf_snapshot", snapshot)
    manifest = assets.prefetch_models(config, tmp_path, ["kumo"])
    assert calls == [("nvidia/Kumo-Tabular", "v1.0.1")]
    assert manifest["tabpfn"] == original_tabpfn
    frozen_bytes = (cache / "model_manifest.json").read_bytes()
    assert assets.prefetch_models(config, tmp_path, ["kumo"]) == manifest
    assert len(calls) == 1
    assert (cache / "model_manifest.json").read_bytes() == frozen_bytes
    (tmp_path / config["models"]["kumo"]["checkpoint_dir"] / "large" / "regressor.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        assets.prefetch_models(config, tmp_path, ["kumo"])
    assert len(calls) == 1
    assert (cache / "model_manifest.json").read_bytes() == frozen_bytes


def test_prefetch_rejects_incompatible_frozen_repository_without_network(tmp_path, monkeypatch):
    config = fixture_config()
    cache = tmp_path / "cache"
    cache.mkdir()
    atomic_json(cache / "model_manifest.json", {"kumo": {"checkpoint": {
        "repo_id": "wrong/model", "path": str(tmp_path / "cache/models/kumo")}}})
    monkeypatch.setattr(assets, "hf_snapshot", lambda *a, **kw: pytest.fail("must not download"))
    with pytest.raises(ValueError, match="configured repository/path"):
        assets.prefetch_models(config, tmp_path, ["kumo"])


def test_ortholog_snapshot_is_verified_and_reused_without_regeneration(tmp_path, monkeypatch):
    calls = []

    def download(url, destination):
        calls.append(url)
        destination.write_text("MouseA\tHUMANA\tortholog_one2one\tENSM1\tENSG1\n")
        return {"path": str(destination), "url": url, "sha256": file_sha256(destination)}

    monkeypatch.setattr(assets, "download", download)
    destination = tmp_path / "mapping.tsv"
    record = assets.fetch_orthologs(destination)
    assert record["pairs"] == 1
    assert record["gene_ids_sha256"] == file_sha256(tmp_path / "mouse_gene_ids.tsv")
    assert assets.fetch_orthologs(destination) == record
    assert len(calls) == 1
    (tmp_path / "mouse_gene_ids.tsv").write_text("changed")
    with pytest.raises(ValueError, match="missing or changed"):
        assets.fetch_orthologs(destination)
    assert len(calls) == 1


def test_extracted_uce_resources_are_verified_against_their_archive_directory(tmp_path):
    proteins = tmp_path / "protein_embeddings"
    proteins.mkdir()
    (proteins / "mouse.pt").write_bytes(b"synthetic")
    archive = tmp_path / "protein_embeddings.tar.gz"
    archive.write_bytes(b"archive-fixture")
    record = {"files": [{"path": str(archive), "sha256": file_sha256(archive)}],
              "protein_files": [{"name": "mouse.pt", "sha256": file_sha256(proteins / "mouse.pt")}]}
    assets.verify_resource_record(record)
    (proteins / "mouse.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        assets.verify_resource_record(record)
