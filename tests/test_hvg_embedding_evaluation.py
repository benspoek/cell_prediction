"""Fold embedding evaluation/reporting integration without model downloads."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cell_property_prediction import evaluation as ev
from cell_property_prediction import hvg_embeddings as he
from cell_property_prediction.config import atomic_json, file_sha256
from cell_property_prediction.data import LOG_TARGETS
from cell_property_prediction.reporting import report


@pytest.fixture
def fold_benchmark(tmp_path, monkeypatch):
    ad = pytest.importorskip("anndata")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    splits = tmp_path / "splits"
    splits.mkdir()
    ids = [f"cell{i}" for i in range(12)]
    genes = ["G0", "G1", "G2", "G3"]
    counts = np.arange(1, 49, dtype=np.float32).reshape(12, 4)
    ad.AnnData(sparse.csr_matrix(counts), obs=pd.DataFrame(index=ids),
               var=pd.DataFrame(index=genes)).write_h5ad(prepared / "counts.h5ad")
    np.savez(prepared / "expression.npz", values=np.log1p(counts),
             cell_ids=np.array(ids), gene_names=np.array(genes))
    x = np.arange(12, dtype=float)
    targets = pd.DataFrame({"AP threshold (mV)": -60 + x,
                            LOG_TARGETS[0]: np.exp(0.1 * x)}, index=pd.Index(ids, name="cell_id"))
    targets.to_csv(prepared / "targets_raw.csv")
    pd.DataFrame({"mouse": [f"m{i // 2}" for i in range(12)]},
                 index=targets.index).to_csv(prepared / "metadata.csv")
    folds = [{"train_ids": ids[:6], "test_ids": ids[6:]},
             {"train_ids": ids[6:], "test_ids": ids[:6]}]
    atomic_json(splits / "cell_kfold.json", {"schema_version": 1, "scheme": "cell_kfold",
                "cell_ids": ids, "n_splits": 2, "never_tested_ids": [], "folds": folds})
    config = {"seed": 42, "paths": {"prepared_dir": "prepared", "splits_dir": "splits",
              "embeddings_dir": "embeddings", "results_dir": "results", "cache_dir": "cache"},
              "evaluation": {"protocols": ["cell_kfold"], "inner_folds": 2,
                             "ridge_alphas": [0.1, 1], "bootstrap_samples": 10},
              "embeddings": {name: {} for name in he.ENCODERS}, "models": {}, "cluster": {}}

    def selector(config, prepared, representation, train_ids, test_ids):
        # Different native training-fold masks, while raw counts remain unchanged.
        selected = genes[:2] if train_ids[0] == ids[0] else genes[2:]
        return None, None, {"genes": selected, "training_ids": list(train_ids)}, {}

    def encoder_context(name, settings, root, source, device):
        return {"encoder": name, "config": dict(settings), "input_sha256": file_sha256(source),
                "device": str(device), "python": "synthetic", "packages": {}}

    def encode(name, data):
        values = np.zeros((data.n_obs, he.DIMENSIONS[name]), dtype=np.float32)
        number = np.array([int(cell.removeprefix("cell")) for cell in data.obs_names])
        values[:, 0] = number
        values[:, 1] = number ** 2
        return values, {"synthetic_encoder": True}

    monkeypatch.setattr(he.hvg, "build_features", selector)
    monkeypatch.setattr(he.embeddings, "_embedding_context", encoder_context)
    monkeypatch.setattr(he, "_coverage", lambda data, *args: {
        "matched_genes": data.n_vars, "all_cells_covered": True,
        "cells_with_positive_in_vocab_expression": data.n_obs})
    monkeypatch.setattr(he.embeddings, "_genept", lambda data, *args: encode("genept", data))
    monkeypatch.setattr(he.embeddings, "_uce", lambda data, *args: encode("uce", data))
    monkeypatch.setattr(he.embeddings, "_nicheformer", lambda data, *args: encode("nicheformer", data))
    he.prepare_hvg_panels(config, tmp_path)
    for fold in range(2):
        for name in he.ENCODERS:
            he.generate_hvg_embedding(name, config, root=tmp_path, protocol="cell_kfold",
                                      fold=fold, device="cpu")
    embeddings = tmp_path / "embeddings"
    np.savez(embeddings / "genept.npz", values=counts[:, :2], cell_ids=np.array(ids))
    atomic_json(embeddings / "genept.audit.json", {
        "output_sha256": file_sha256(embeddings / "genept.npz"), "labels_used": False, "frozen": True})
    return tmp_path, config, ids, folds


def _evaluate_both(root, config, representation="hvg_genept", predictor="mean"):
    return [ev.evaluate_fold(config, root=root, protocol="cell_kfold", representation=representation,
                             predictor=predictor, fold=fold, device="cpu") for fold in range(2)]


@pytest.mark.parametrize("representation", sorted(ev.HVG_EMBEDDING_REPRESENTATIONS))
def test_features_require_exact_outer_context(fold_benchmark, representation):
    root, config, ids, folds = fold_benchmark
    with pytest.raises(ValueError, match="protocol and fold"):
        ev._features(config, root, representation, ids[:6], ids[6:])
    with pytest.raises(ValueError, match="exact outer split"):
        ev._features(config, root, representation, ids[:6], ids[6:], protocol="cell_kfold", fold=1)
    train, query, audit, inputs = ev._features(config, root, representation, ids[:6], ids[6:],
                                             protocol="cell_kfold", fold=0)
    assert train.shape == query.shape == (6, he.DIMENSIONS[representation.removeprefix("hvg_")])
    assert audit["feature_scope"]["fold"] == 0 and len(inputs) == 4
    np.testing.assert_array_equal(train[:, 0], np.arange(6))
    np.testing.assert_array_equal(query[:, 0], np.arange(6, 12))


def test_report_pools_valid_fold_npz_and_legacy_methods(fold_benchmark):
    root, config, ids, folds = fold_benchmark
    manifests = _evaluate_both(root, config)
    _evaluate_both(root, config, representation="genept")
    scopes = [manifest["features"]["feature_scope"] for manifest in manifests]
    assert scopes[0]["stable_context_sha256"] == scopes[1]["stable_context_sha256"]
    assert scopes[0]["fold_artifacts"]["embedding"]["path"] != scopes[1]["fold_artifacts"]["embedding"]["path"]
    assert {Path(path).name for path in manifests[0]["code_sha256"]} >= {
        "hvg_embeddings.py", "hvg.py", "rrr.py", "embeddings.py"}
    result = report(config, root)
    assert result["n_methods_complete_oof"] == 2
    assert "raw physical units" in result["primary_metric"]
    assert "authors_joint_r2" in result["secondary_metrics"]
    summary = pd.read_csv(root / "results/reports/summary.csv")
    assert set(summary.representation) == {"hvg_genept", "genept"}
    assert summary.complete_oof.all() and summary["rank"].notna().all()
    assert summary.authors_joint_r2.notna().all()
    assert (summary.authors_joint_r2_status == "complete").all()
    assert len(pd.read_csv(root / "results/reports/authors_joint_r2_folds.csv")) == 4


@pytest.mark.parametrize("damage", ["wrong_fold", "mask", "embedding", "stable_context"])
def test_report_rejects_altered_fold_scope_or_artifacts(fold_benchmark, damage):
    root, config, ids, folds = fold_benchmark
    _evaluate_both(root, config)
    manifest_path = root / "results/cell_kfold/hvg_genept/mean/fold_00/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    scope = manifest["features"]["feature_scope"]
    if damage == "wrong_fold":
        scope["fold"] = 1
        atomic_json(manifest_path, manifest)
    elif damage == "mask":
        selection_path = Path(scope["fold_artifacts"]["selection"]["path"])
        selection = json.loads(selection_path.read_text())
        selection["genes"] = list(reversed(selection["genes"]))
        atomic_json(selection_path, selection)
    elif damage == "embedding":
        with Path(scope["fold_artifacts"]["embedding"]["path"]).open("ab") as handle:
            handle.write(b"changed NPZ")
    else:
        scope["stable_context"]["seed"] = 43
        scope["stable_context_sha256"] = he._digest(scope["stable_context"])
        atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="scope|context|checksum|genes|artifact|altered"):
        report(config, root)


@pytest.mark.parametrize("context", ["input_sha256", "model_input_sha256", "code_sha256"])
def test_report_preserves_common_input_model_code_consistency(fold_benchmark, context):
    root, config, ids, folds = fold_benchmark
    _evaluate_both(root, config)
    manifest_path = root / "results/cell_kfold/hvg_genept/mean/fold_01/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    additional = root / "additional_context.txt"
    additional.write_text("different context")
    manifest[context][str(additional)] = file_sha256(additional)
    atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="contexts differ"):
        report(config, root)


@pytest.mark.parametrize("predictor", ["tabpfn", "limix", "kumo"])
@pytest.mark.parametrize("representation", sorted(ev.HVG_EMBEDDING_REPRESENTATIONS))
def test_predictor_adapters_receive_fold_embeddings(fold_benchmark, monkeypatch, predictor, representation):
    root, config, ids, folds = fold_benchmark
    from cell_property_prediction import models
    calls = []

    class Predictor:
        def predict(self, train, y, query, run_id):
            calls.append((train.copy(), query.copy(), run_id))
            return np.zeros(len(query)), {"synthetic": True}

    monkeypatch.setattr(models, "build_predictor", lambda *args, **kwargs: Predictor())
    manifest = ev.evaluate_fold(config, root=root, protocol="cell_kfold", representation=representation,
                                predictor=predictor, fold=0, device="cpu")
    assert len(calls) == 2
    assert manifest["features"]["feature_scope"]["encoder"] == representation.removeprefix("hvg_")
    for train, query, run_id in calls:
        assert train.shape[1] == query.shape[1] == he.DIMENSIONS[representation.removeprefix("hvg_")]
        np.testing.assert_array_equal(train[:, 0], np.arange(6))
        np.testing.assert_array_equal(query[:, 0], np.arange(6, 12))
        assert representation in run_id and predictor in run_id


def test_ridge_requires_separate_inner_fold_encodings(fold_benchmark):
    root, config, ids, folds = fold_benchmark
    with pytest.raises(ValueError, match="separate inner-fold selection and encoding"):
        ev.evaluate_fold(config, root=root, protocol="cell_kfold", representation="hvg_genept",
                         predictor="ridge", fold=0, device="cpu")
