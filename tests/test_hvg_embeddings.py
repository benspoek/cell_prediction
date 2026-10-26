"""Synthetic-only native-HVG encoder tests; no model downloads or real inference."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cell_property_prediction import embeddings, hvg
from cell_property_prediction import hvg_embeddings as he
from cell_property_prediction.config import atomic_json, file_sha256


class FakeAnnData:
    """Minimal on-disk counts fixture; optional real AnnData is tested below."""
    def __init__(self, X, obs, var, layers=None):
        self.X = sparse.csr_matrix(X)
        self.obs, self.var = obs.copy(), var.copy()
        self.layers = dict(layers or {})
        self.raw = None
        self.uns, self.obsm, self.varm = {}, {}, {}

    @property
    def obs_names(self):
        return self.obs.index

    @property
    def var_names(self):
        return self.var.index

    @property
    def n_obs(self):
        return self.X.shape[0]

    @property
    def n_vars(self):
        return self.X.shape[1]

    def __getitem__(self, axes):
        rows, columns = axes
        rows = self.obs_names.get_indexer(rows)
        columns = self.var_names.get_indexer(columns)
        if (rows < 0).any() or (columns < 0).any():
            raise KeyError("Missing fixture labels")
        return FakeAnnData(self.X[rows][:, columns], self.obs.iloc[rows], self.var.iloc[columns],
                           {key: value[rows][:, columns] for key, value in self.layers.items()})

    def write_h5ad(self, path):
        import pickle
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def read_h5ad(cls, path, **kwargs):
        import pickle
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


@pytest.fixture
def panel_fixture(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "anndata", types.SimpleNamespace(
        AnnData=FakeAnnData, read_h5ad=FakeAnnData.read_h5ad))
    return _fixture_data(tmp_path)


def _fixture_data(root, adata_type=FakeAnnData):
    prepared, splits = root / "prepared", root / "splits"
    prepared.mkdir()
    splits.mkdir()
    cells = [f"cell{i:02d}" for i in range(48)]
    genes = ["VariableA", "VariableB", "Constant", "Ubiquitous"]
    exons = np.zeros((48, 4), dtype=np.float32)
    for offset in (0, 24):
        exons[offset:offset + 12, 0] = 200
        exons[offset:offset + 18, 1] = 100
        exons[offset:offset + 12, 2] = 200
    exons[:, 3] = 300
    position = np.arange(48, dtype=np.float32)
    expression = np.column_stack([position + 1, (position % 7) + 1,
                                  np.ones(48), position + 2])
    np.savez(prepared / "expression.npz", values=expression,
             cell_ids=np.asarray(cells), gene_names=np.asarray(genes))
    # Deliberately reverse original counts axes and include forbidden labels;
    # panels must explicitly align both axes and drop all labels/layers.
    full = exons + 1000
    counts = adata_type(X=sparse.csr_matrix(full[::-1, ::-1]),
        obs=pd.DataFrame({"species": "mouse", "outcome": np.arange(48)}, index=cells[::-1]),
        var=pd.DataFrame({"label": "forbidden"}, index=genes[::-1]),
        layers={"exon_counts": sparse.csr_matrix(exons[::-1, ::-1])})
    counts.write_h5ad(prepared / "counts.h5ad")
    folds = [{"train_ids": cells[24:][::-1], "test_ids": cells[:24][::-1]},
             {"train_ids": cells[:24], "test_ids": cells[24:]}]
    protocols = ["cell_kfold", "mouse_group_kfold"]
    for protocol in protocols:
        atomic_json(splits / f"{protocol}.json", {"schema_version": 1,
            "scheme": protocol, "cell_ids": cells, "n_splits": 2,
            "folds": folds, "never_tested_ids": []})
    (root / "resource.bin").write_bytes(b"frozen synthetic encoder resources")
    config = {"seed": 42, "paths": {"prepared_dir": "prepared", "splits_dir": "splits",
              "embeddings_dir": "embeddings"}, "evaluation": {"protocols": protocols},
              "embeddings": {name: {"seed": 42} for name in he.ENCODERS},
              "cluster": {"remaining_gpu_hours": 100, "encoder_cpus": 8}}
    return root, config, cells, genes, full, folds


@pytest.fixture
def mocked_encoders(panel_fixture, monkeypatch):
    root, config, cells, genes, full, folds = panel_fixture
    he.prepare_hvg_panels(config, root)
    runtime = {"python": "encoder Python 3.11", "packages": {"numpy": "encoder-version"}}
    seen = []

    def context(name, settings, context_root, path, device):
        return {"schema_version": 1, "embedding": name, "config": dict(settings),
                "prepared_sha256": he._sha(path), "implementation_sha256": he._sha(embeddings.__file__),
                "device": device, **runtime,
                "resources": {str(root / "resource.bin"): he._sha(root / "resource.bin")}}

    def encode(name, data):
        ids = data.obs_names.astype(str).tolist()
        seen.append((name, ids, data.var_names.astype(str).tolist(), data.X.toarray(),
                     list(data.obs), list(data.var), list(data.layers)))
        values = np.repeat(np.asarray([int(cell.removeprefix("cell")) for cell in ids], dtype=np.float32)[:, None],
                           he.DIMENSIONS[name], axis=1)
        return values, {"mocked_encoder": True}

    monkeypatch.setattr(embeddings, "_embedding_context", context)
    monkeypatch.setattr(embeddings, "_genept", lambda data, *args: encode("genept", data))
    monkeypatch.setattr(embeddings, "_nicheformer", lambda data, *args: encode("nicheformer", data))
    monkeypatch.setattr(embeddings, "_uce", lambda data, *args: encode("uce", data))
    monkeypatch.setattr(he, "_coverage", lambda data, *args: {
        "matched_genes": data.n_vars, "cells_with_positive_in_vocab_expression": data.n_obs,
        "min_in_vocab_counts": 1, "all_cells_covered": True})
    return panel_fixture, runtime, seen


def test_panels_use_exact_native_genes_and_raw_counts(panel_fixture):
    root, config, cells, genes, full, folds = panel_fixture
    result = he.prepare_hvg_panels(config, root)
    assert result["n_panels"] == 4
    assert he.check_hvg_panels(config, root)["valid"]
    directory = root / "embeddings/hvg/cell_kfold/fold_00"
    selection = json.loads((directory / "selection.json").read_text())
    _, _, native, _ = hvg.build_features(config, root / "prepared", "hvg",
                                        folds[0]["train_ids"], folds[0]["test_ids"])
    assert selection["selector_audit"] == native
    assert selection["genes"] == ["VariableA", "VariableB"]
    assert native["zero_variance_genes_removed"] == ["Constant"]
    data = FakeAnnData.read_h5ad(directory / "counts.h5ad")
    assert data.obs_names.tolist() == folds[0]["train_ids"] + folds[0]["test_ids"]
    expected_rows = [cells.index(cell) for cell in data.obs_names]
    np.testing.assert_array_equal(data.X.toarray(), full[expected_rows, :2])
    assert list(data.obs) == ["species"] and not list(data.var) and not data.layers
    hashes = {p: file_sha256(p) for p in directory.iterdir()}
    assert all(row["reused"] for row in he.prepare_hvg_panels(config, root)["panels"])
    assert hashes == {p: file_sha256(p) for p in directory.iterdir()}


@pytest.mark.parametrize("name", he.ENCODERS)
def test_all_encoders_get_training_first_counts_save_canonical_ids(mocked_encoders, name):
    fixture, _, seen = mocked_encoders
    root, config, cells, genes, full, folds = fixture
    result = he.generate_hvg_embedding(name, config, root=root, protocol="cell_kfold", fold=0, device="cuda")
    assert result.path.parent == root / "embeddings/hvg/cell_kfold/fold_00"
    _, observed, selected, counts, obs, var, layers = seen[-1]
    assert observed == folds[0]["train_ids"] + folds[0]["test_ids"]
    assert selected == ["VariableA", "VariableB"]
    np.testing.assert_array_equal(counts, full[[cells.index(cell) for cell in observed], :2])
    assert obs == ["species"] and not var and not layers
    with np.load(result.path, allow_pickle=False) as archive:
        assert archive["cell_ids"].tolist() == cells
        assert archive["values"].shape == (48, he.DIMENSIONS[name])
        np.testing.assert_array_equal(archive["values"][:, 0], np.arange(48))
    original_hash = file_sha256(result.path)
    assert he.generate_hvg_embedding(name, config, root=root, protocol="cell_kfold", fold=0,
                                     device="cuda").audit == result.audit
    assert len(seen) == 1 and original_hash == file_sha256(result.path)
    train, query, audit, inputs = he.load_hvg_features(config, root, f"hvg_{name}", "cell_kfold", 0,
                                                      folds[0]["train_ids"], folds[0]["test_ids"])
    assert train.shape[1] == query.shape[1] == he.DIMENSIONS[name]
    assert set(audit["feature_scope"]["fold_artifacts"]) == {"selection", "counts", "embedding", "audit"}
    assert {Path(path).name for path in inputs} == {"selection.json", "counts.h5ad", f"{name}.npz", f"{name}.audit.json"}
    assert he.validate_hvg_feature_scope(config, root, audit, protocol="cell_kfold", fold=0,
        train_ids=folds[0]["train_ids"], test_ids=folds[0]["test_ids"]) == result.audit["stable_context_sha256"]


def test_fold_family_context_and_consumer_environment(mocked_encoders):
    fixture, runtime, _ = mocked_encoders
    root, config, _, _, _, folds = fixture
    first = he.generate_hvg_embedding("uce", config, root=root, protocol="cell_kfold", fold=0, device="cuda")
    second = he.generate_hvg_embedding("uce", config, root=root, protocol="cell_kfold", fold=1, device="cuda")
    assert first.audit["stable_context_sha256"] == second.audit["stable_context_sha256"]
    assert first.audit["panel_context_sha256"] != second.audit["panel_context_sha256"]
    runtime.update(python="consumer Python 3.12", packages={"numpy": "consumer-version"})
    _, _, audit, _ = he.load_hvg_features(config, root, "hvg_uce", "cell_kfold", 0,
                                         folds[0]["train_ids"], folds[0]["test_ids"])
    he.validate_hvg_feature_scope(config, root, audit, protocol="cell_kfold", fold=0,
                                  train_ids=folds[0]["train_ids"], test_ids=folds[0]["test_ids"])
    with pytest.raises(ValueError, match="context"):
        he.generate_hvg_embedding("uce", config, root=root, protocol="cell_kfold", fold=0, device="cuda")


@pytest.mark.parametrize("change", ["resource", "config", "split", "source", "audit", "embedding"])
def test_loader_rejects_changed_provenance(mocked_encoders, monkeypatch, change):
    fixture, _, _ = mocked_encoders
    root, config, _, _, _, folds = fixture
    result = he.generate_hvg_embedding("genept", config, root=root, protocol="cell_kfold", fold=0, device="cpu")
    _, _, saved, _ = he.load_hvg_features(config, root, "hvg_genept", "cell_kfold", 0,
                                         folds[0]["train_ids"], folds[0]["test_ids"])
    if change == "resource":
        (root / "resource.bin").write_bytes(b"changed frozen resource")
    elif change == "config":
        config["embeddings"]["genept"]["seed"] = 999
    elif change == "split":
        path = root / "splits/cell_kfold.json"
        path.write_text(path.read_text() + " ")
    elif change == "source":
        original = he._sources()
        monkeypatch.setattr(he, "_sources", lambda: {**original, "hvg_embeddings.py": "changed"})
    elif change == "audit":
        path = result.path.with_name("genept.audit.json")
        path.write_text(path.read_text() + " ")
    else:
        result.path.write_bytes(b"changed archive")
    with pytest.raises(ValueError):
        he.validate_hvg_feature_scope(config, root, saved, protocol="cell_kfold", fold=0,
                                      train_ids=folds[0]["train_ids"], test_ids=folds[0]["test_ids"])


@pytest.mark.parametrize("change", ["fold", "ids", "partial", "genes"])
def test_panel_scope_or_partial_artifacts_are_rejected(panel_fixture, change):
    root, config, _, _, _, _ = panel_fixture
    he.prepare_hvg_panels(config, root)
    path = root / "embeddings/hvg/cell_kfold/fold_00/selection.json"
    payload = json.loads(path.read_text())
    if change == "fold":
        payload["fold"] = 1
    elif change == "ids":
        payload["panel_context"]["train_ids"] = payload["panel_context"]["train_ids"][::-1]
    elif change == "genes":
        payload["genes"] = ["Constant"]
    else:
        path.unlink()
    if change != "partial":
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        he.check_hvg_panels(config, root)


def test_genept_coverage_checks_positive_expression_for_gene_count_pooling(panel_fixture):
    root, config, _, _, _, _ = panel_fixture
    vectors = root / "vectors.npz"
    np.savez(vectors, gene_symbols=np.asarray(["HUMAN_A"]), vectors=np.ones((1, 3072)))
    orthologs = root / "orthologs.tsv"
    orthologs.write_text("mouse_symbol\thuman_symbol\nVariableA\tHUMAN_A\n")
    config["embeddings"]["genept"] = {"vectors_path": str(vectors), "orthologs_path": str(orthologs),
                                      "pooling": "gene_count_mean"}
    data = FakeAnnData(sparse.csr_matrix([[3, 0], [0, 4]]),
                      pd.DataFrame({"species": "mouse"}, index=["a", "b"]),
                      pd.DataFrame(index=["VariableA", "VariableB"]))
    with pytest.raises(ValueError, match="positive in-vocabulary"):
        he._coverage(data, "genept", config, root)
    data.X = sparse.csr_matrix([[3, 0], [2, 4]])
    assert he._coverage(data, "genept", config, root)["all_cells_covered"]


def test_hash_cache_uses_file_stat_and_invalidates_on_change(tmp_path, monkeypatch):
    path = tmp_path / "large-resource"
    path.write_bytes(b"first")
    calls = []
    original = he.file_sha256
    monkeypatch.setattr(he, "file_sha256", lambda value: (calls.append(value), original(value))[1])
    first = he._sha(path)
    assert he._sha(path) == first and len(calls) == 1
    path.write_bytes(b"second longer content")
    assert he._sha(path) != first and len(calls) == 2


def test_budget_uses_cuda_pilots_safety_margin_and_actual_remaining_jobs(mocked_encoders, monkeypatch):
    fixture, _, _ = mocked_encoders
    root, config, _, _, _, _ = fixture
    for name in he.ENCODERS:
        result = he.generate_hvg_embedding(name, config, root=root, protocol="cell_kfold", fold=0,
                                           device="cpu" if name == "genept" else "cuda")
        result.audit["elapsed_seconds"] = 3600
        result.path.with_name(f"{name}.audit.json").write_text(json.dumps(result.audit))
    budget = he.encoding_budget(config, root)
    assert budget["remaining_jobs_per_encoder"] == dict.fromkeys(he.ENCODERS, 3)
    assert budget["remaining_gpu_hours_estimate"] == 12
    assert budget["remaining_cpu_wall_hours_estimate"] == 6
    assert budget["remaining_cpu_core_hours_estimate"] == 48
    assert budget["approved"]
    config["cluster"]["remaining_gpu_hours"] = 11
    assert not he.encoding_budget(config, root)["approved"]
    path = root / "embeddings/hvg/cell_kfold/fold_00/uce.audit.json"
    audit = json.loads(path.read_text())
    audit["device"] = "cpu"
    audit["stable_context"]["encoder_context"]["device"] = "cpu"
    audit["stable_context_sha256"] = he._digest(audit["stable_context"])
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="CUDA-measured"):
        he.encoding_budget(config, root)


def test_actual_anndata_counts_panel_integration(tmp_path):
    # Optional on Windows; all mandatory tests use synthetic in-process I/O.
    ad = pytest.importorskip("anndata")
    root, config, cells, genes, full, folds = _fixture_data(tmp_path, ad.AnnData)
    he.prepare_hvg_panels(config, root)
    data = ad.read_h5ad(root / "embeddings/hvg/cell_kfold/fold_00/counts.h5ad")
    assert data.obs_names.tolist() == folds[0]["train_ids"] + folds[0]["test_ids"]
    assert list(data.obs) == ["species"] and not list(data.var) and not data.layers
    assert he.check_hvg_panels(config, root)["valid"]


@pytest.mark.parametrize("bad", ["ids", "nan", "dimensions"])
def test_embedding_loader_validates_arrays_even_with_updated_checksums(mocked_encoders, bad):
    fixture, _, _ = mocked_encoders
    root, config, _, _, _, folds = fixture
    result = he.generate_hvg_embedding("genept", config, root=root, protocol="cell_kfold", fold=0, device="cpu")
    with np.load(result.path, allow_pickle=False) as archive:
        values, ids = archive["values"].copy(), archive["cell_ids"].copy()
    if bad == "ids":
        ids = ids[::-1]
    elif bad == "nan":
        values[0, 0] = np.nan
    else:
        values = values[:, :10]
    with result.path.open("wb") as handle:
        np.savez(handle, values=values, cell_ids=ids)
    result.audit["output_sha256"] = file_sha256(result.path)
    result.path.with_name("genept.audit.json").write_text(json.dumps(result.audit))
    with pytest.raises(ValueError, match="finite dimensions or canonical"):
        he.load_hvg_features(config, root, "hvg_genept", "cell_kfold", 0,
                             folds[0]["train_ids"], folds[0]["test_ids"])


def test_uce_coverage_uses_native_mouse_filename_and_uppercase_mapping(panel_fixture, monkeypatch):
    root, config, _, _, _, _ = panel_fixture
    resources = root / "uce-resources"
    resources.mkdir()
    pd.DataFrame({"species": ["mouse", "human"], "gene_symbol": ["VARIABLEA", "VARIABLEB"],
                  "chromosome": [1, 2]}).to_csv(resources / "species_chrom.csv", index=False)
    config["embeddings"]["uce"]["resources_dir"] = str(resources)
    loaded = []
    def protein(path, **kwargs):
        loaded.append(Path(path).name)
        return {"VariableA": np.ones(2), "VariableB": np.ones(2)}
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(load=protein))
    data = FakeAnnData(sparse.csr_matrix([[3, 0], [0, 4]]),
                      pd.DataFrame({"species": "mouse"}, index=["a", "b"]),
                      pd.DataFrame(index=["VariableA", "VariableB"]))
    with pytest.raises(ValueError, match="positive in-vocabulary"):
        he._coverage(data, "uce", config, root)
    data.X = sparse.csr_matrix([[3, 0], [2, 4]])
    assert he._coverage(data, "uce", config, root)["all_cells_covered"]
    assert loaded == ["Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt"] * 2


def test_nicheformer_coverage_checks_mapped_native_genes(panel_fixture):
    root, config, _, _, _, _ = panel_fixture
    checkpoint = root / "nicheformer"
    checkpoint.mkdir()
    FakeAnnData(sparse.csr_matrix([[1]]), pd.DataFrame(index=["reference"]),
                pd.DataFrame(index=["native-id"])).write_h5ad(checkpoint / "model.h5ad")
    mapping = root / "mapping.tsv"
    mapping.write_text("mouse_symbol\tmodel_gene_id\nVariableA\tnative-id\nVariableB\tunknown\n")
    config["embeddings"]["nicheformer"].update(checkpoint_dir=str(checkpoint), gene_mapping_path=str(mapping))
    data = FakeAnnData(sparse.csr_matrix([[3, 0], [0, 4]]),
                      pd.DataFrame({"species": "mouse"}, index=["a", "b"]),
                      pd.DataFrame(index=["VariableA", "VariableB"]))
    with pytest.raises(ValueError, match="positive in-vocabulary"):
        he._coverage(data, "nicheformer", config, root)
    data.X = sparse.csr_matrix([[3, 0], [2, 4]])
    assert he._coverage(data, "nicheformer", config, root)["matched_genes"] == 1


def test_encoding_failure_cannot_leave_partial_published_output(mocked_encoders, monkeypatch):
    fixture, _, _ = mocked_encoders
    root, config, _, _, _, _ = fixture
    def bad(data, *args):
        return np.full((data.n_obs, 3072), np.nan), {}
    monkeypatch.setattr(embeddings, "_genept", bad)
    with pytest.raises(ValueError, match="finite native"):
        he.generate_hvg_embedding("genept", config, root=root, protocol="cell_kfold", fold=0, device="cpu")
    output, audit = he._embedding_paths(config, root, "genept", "cell_kfold", 0)
    assert not output.exists() and not audit.exists()
    output.write_bytes(b"interrupted partial output")
    with pytest.raises(ValueError, match="partial immutable"):
        he.generate_hvg_embedding("genept", config, root=root, protocol="cell_kfold", fold=0, device="cpu")
