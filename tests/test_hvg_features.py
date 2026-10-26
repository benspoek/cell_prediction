"""Synthetic checks for author-style HVG inputs, without model or data downloads."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cell_property_prediction import evaluation as ev
from cell_property_prediction import hvg
from cell_property_prediction.config import file_sha256
from cell_property_prediction.rrr import gene_selection


@pytest.fixture
def hvg_data(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    ids = [f"cell{i}" for i in range(36)]
    genes = np.array(["VariableA", "VariableB", "Ubiquitous", "TestOnly", "Constant", "AtThreshold"])
    exon = np.zeros((36, 6), dtype=np.float32)
    exon[:12, 0], exon[:18, 1], exon[:, 2] = 200, 100, 300
    exon[24:, 3], exon[:12, 4], exon[:, 5] = 300, 200, 32
    position = np.arange(36, dtype=np.float32)
    expression = np.column_stack([position + 1, (position % 7) + 1,
                                  position + 2, position + 3, np.ones(36), position + 4])
    np.savez(prepared / "expression.npz", values=expression, cell_ids=np.array(ids), gene_names=genes)
    (prepared / "counts.h5ad").write_bytes(b"synthetic exon layer")
    monkeypatch.setattr(hvg, "_load_exon_counts", lambda *args: sparse.csr_matrix(exon))
    config = {"seed": 42, "paths": {"prepared_dir": "prepared", "splits_dir": "splits",
              "results_dir": "results", "cache_dir": "cache"},
              "evaluation": {"protocols": ["cell_kfold", "mouse_group_kfold"],
                             "minimum_gene_cells": 999, "pca_components": 7}, "models": {}}
    return tmp_path, config, ids, genes, exon, expression


def test_hvg_and_hvg_pca_use_the_same_author_selected_genes(hvg_data):
    root, config, ids, genes, exon, _ = hvg_data
    x, query, audit, inputs = ev._features(config, root, "hvg", ids[:24], ids[24:])
    pca_x, pca_query, pca_audit, _ = ev._features(config, root, "hvg_pca128", ids[:24], ids[24:])
    expected_boundary = gene_selection(sparse.csr_matrix(exon[:24]), **hvg.HVG_PARAMETERS)
    assert genes[expected_boundary].tolist() == ["VariableA", "VariableB", "Constant"]
    assert audit["n_boundary_selected_genes"] == 3
    assert audit["zero_variance_genes_removed"] == ["Constant"]
    assert audit["genes"] == pca_audit["genes"] == ["VariableA", "VariableB"]
    assert x.shape == pca_x.shape == (24, 2)
    assert query.shape == pca_query.shape == (12, 2)
    np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(x.std(axis=0), 1, atol=1e-6)
    assert audit["selection_parameters"] == {"n": 1000, "threshold": 32, "atleast": 10,
                                              "decay": 1.5, "yoffset": .02, "xoffset": 5}
    assert audit["fit_scope"] == "training_cells_only" and not audit["labels_used"]
    assert pca_audit["pca_requested_components"] == 128
    assert pca_audit["pca_components"] == 2  # ignore unrelated broad-PCA config override
    assert inputs[str(root / "prepared/counts.h5ad")] == file_sha256(root / "prepared/counts.h5ad")
    assert inputs[str(root / "prepared/expression.npz")] == file_sha256(root / "prepared/expression.npz")


@pytest.mark.parametrize("representation", ["hvg", "hvg_pca128"])
def test_held_out_expression_and_exons_cannot_change_training_features(hvg_data, representation):
    root, config, ids, genes, exon, expression = hvg_data
    x, query, audit, _ = ev._features(config, root, representation, ids[:24], ids[24:])
    exon[24:] = 1e8
    expression[24:] = 1e5 * np.arange(1, 7)
    np.savez(root / "prepared/expression.npz", values=expression, cell_ids=np.array(ids), gene_names=genes)
    changed_x, changed_query, changed_audit, _ = ev._features(config, root, representation, ids[:24], ids[24:])
    np.testing.assert_array_equal(x, changed_x)
    assert audit == changed_audit
    assert not np.allclose(query, changed_query)


def test_hvg_rejects_missing_counts_and_unknown_split_cells(hvg_data):
    root, config, ids, _, _, _ = hvg_data
    with pytest.raises(ValueError, match="lack 1 split cells"):
        ev._features(config, root, "hvg", ids[:24], ["unknown"])
    (root / "prepared/counts.h5ad").unlink()
    with pytest.raises(FileNotFoundError, match="raw exon counts"):
        ev._features(config, root, "hvg", ids[:24], ids[24:])


def _fake_h5ad(monkeypatch, values, cells, genes, *, layer=True):
    annotated = types.SimpleNamespace(obs_names=pd.Index(cells), var_names=pd.Index(genes),
        layers={"exon_counts": sparse.csr_matrix(values)} if layer else {})
    monkeypatch.setitem(sys.modules, "anndata", types.SimpleNamespace(read_h5ad=lambda path: annotated))


def test_exon_loader_realigns_both_identifiers_and_uses_the_exon_layer(monkeypatch):
    cells, genes = np.array(["a", "b", "c"]), np.array(["g1", "g2"])
    expected = np.array([[1, 2], [3, 4], [5, 6]])
    _fake_h5ad(monkeypatch, expected[::-1, ::-1], cells[::-1], genes[::-1])
    actual = hvg._load_exon_counts(Path("counts.h5ad"), cells, genes)
    np.testing.assert_array_equal(actual.toarray(), expected)


@pytest.mark.parametrize("case,match", [("missing_layer", "exon_counts layer"),
    ("wrong_cell", "identifiers differ"), ("wrong_gene", "identifiers differ"),
    ("duplicate_cell", "duplicate"), ("duplicate_gene", "duplicate"),
    ("negative", "finite and nonnegative"), ("nan", "finite and nonnegative")])
def test_exon_loader_rejects_bad_identifiers_and_counts(monkeypatch, case, match):
    cells, genes = np.array(["a", "b"]), np.array(["g1", "g2"])
    observed_cells, observed_genes = cells.copy(), genes.copy()
    values = np.array([[1, 2], [3, 4]], dtype=float)
    if case == "wrong_cell": observed_cells[1] = "c"
    if case == "wrong_gene": observed_genes[1] = "g3"
    if case == "duplicate_cell": observed_cells[1] = "a"
    if case == "duplicate_gene": observed_genes[1] = "g1"
    if case == "negative": values[0, 0] = -1
    if case == "nan": values[0, 0] = np.nan
    _fake_h5ad(monkeypatch, values, observed_cells, observed_genes, layer=case != "missing_layer")
    with pytest.raises(ValueError, match=match):
        hvg._load_exon_counts(Path("counts.h5ad"), cells, genes)


@pytest.mark.parametrize("representation", ["hvg", "hvg_pca128"])
@pytest.mark.parametrize("predictor", ["tabpfn", "limix", "kumo"])
@pytest.mark.parametrize("protocol", ["cell_kfold", "mouse_group_kfold"])
def test_both_hvg_inputs_reach_all_models_and_protocols(hvg_data, monkeypatch, representation, predictor, protocol):
    root, config, ids, _, _, _ = hvg_data
    split_dir = root / "splits"
    split_dir.mkdir()
    (split_dir / f"{protocol}.json").write_text(json.dumps({"folds": [
        {"train_ids": ids[:24], "test_ids": ids[24:]}]}))
    targets = pd.DataFrame({"AP threshold (mV)": np.arange(36) - 60}, index=ids)
    targets.to_csv(root / "prepared/targets_raw.csv")
    pd.DataFrame({"mouse": [f"mouse{i // 3}" for i in range(36)]}, index=ids).to_csv(
        root / "prepared/metadata.csv")
    seen = []
    class Spy:
        def predict(self, train, y, query, run_id):
            seen.append((train.shape, query.shape))
            return np.zeros(len(query)), {"synthetic": True}
    def build(name, *args, **kwargs):
        assert name == predictor
        return Spy()
    monkeypatch.setitem(sys.modules, "cell_property_prediction.models", types.SimpleNamespace(build_predictor=build))
    manifest = ev.evaluate_fold(config, root=root, protocol=protocol, representation=representation,
                                predictor=predictor, fold=0, device="cpu")
    assert seen == [((24, 2), (12, 2))]
    assert manifest["features"]["genes"] == ["VariableA", "VariableB"]
    assert {Path(path).name for path in manifest["code_sha256"]} >= {"hvg.py", "rrr.py"}
    assert {Path(path).name for path in manifest["input_sha256"]} >= {"counts.h5ad", "expression.npz"}


def test_h5ad_exon_layer_integration(tmp_path):
    ad = pytest.importorskip("anndata")
    cells, genes = np.array(["a", "b"]), np.array(["g1", "g2"])
    exons = np.array([[0, 44], [99, 0]], dtype=np.float32)
    annotated = ad.AnnData(X=sparse.csr_matrix(exons + 1000),
        obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes),
        layers={"exon_counts": sparse.csr_matrix(exons)})
    path = tmp_path / "counts.h5ad"
    annotated.write_h5ad(path)
    np.testing.assert_array_equal(hvg._load_exon_counts(path, cells, genes).toarray(), exons)


def test_report_adds_hvg_methods_without_changing_base_representation_list(hvg_data):
    from cell_property_prediction.reporting import report

    root, config, ids, _, exon, _ = hvg_data
    config["evaluation"].update({"protocols": ["cell_kfold"], "representations": ["pca128"],
                                 "minimum_gene_cells": 2, "bootstrap_samples": 20})
    # Give each half enough raw-exon detections to select VariableA independently.
    exon[18:, 0] = exon[:18, 0]
    split_dir = root / "splits"
    split_dir.mkdir()
    (split_dir / "cell_kfold.json").write_text(json.dumps({"folds": [
        {"train_ids": ids[:18], "test_ids": ids[18:]},
        {"train_ids": ids[18:], "test_ids": ids[:18]}]}))
    pd.DataFrame({"AP threshold (mV)": np.arange(36) - 60}, index=ids).to_csv(
        root / "prepared/targets_raw.csv")
    pd.DataFrame({"mouse": [f"mouse{i // 3}" for i in range(36)]}, index=ids).to_csv(
        root / "prepared/metadata.csv")
    for representation in ("pca128", "hvg", "hvg_pca128"):
        for fold in range(2):
            manifest = ev.evaluate_fold(config, root=root, protocol="cell_kfold", representation=representation,
                                        predictor="mean", fold=fold, device="cpu")
            if representation == "pca128":
                # Completed baseline methods retain their original source provenance.
                manifest["code_sha256"] = {"evaluation.py": "previous-baseline-source"}
                path = root / f"results/cell_kfold/pca128/mean/fold_{fold:02d}/manifest.json"
                path.write_text(json.dumps(manifest))
    reported = report(config, root)
    assert reported["n_methods_complete_oof"] == 3
    summary = pd.read_csv(root / "results/reports/summary.csv")
    assert set(summary.representation) == {"pca128", "hvg", "hvg_pca128"}
    assert summary.complete_oof.all()
    assert config["evaluation"]["representations"] == ["pca128"]
