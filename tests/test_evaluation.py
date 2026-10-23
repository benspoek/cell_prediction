"""Synthetic tests: no source dataset, foundation-model downloads, or GPU runs."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cell_property_prediction import evaluation as ev
from cell_property_prediction.config import atomic_json, file_sha256
from cell_property_prediction.data import LOG_TARGETS
from cell_property_prediction.reporting import metrics, report


@pytest.fixture
def benchmark(tmp_path):
    prepared, splits, embeddings = (tmp_path / name for name in ("prepared", "splits", "embeddings"))
    for directory in (prepared, splits, embeddings):
        directory.mkdir()
    ids = [f"cell{i}" for i in range(12)]
    x = np.arange(1, 13, dtype=np.float32)
    expression = np.column_stack([x, x ** 0.5, (x % 3) + 1, 2 * x + 1,
                                  np.r_[np.zeros(8), np.arange(1, 5)], np.r_[1, np.zeros(11)]])
    genes = ["Hcn1", "Kcna1", "Cacna1e", "Other", "TestOnly", "Arpp21"]
    np.savez(prepared / "expression.npz", values=expression, cell_ids=np.array(ids), gene_names=np.array(genes))
    targets = pd.DataFrame({"AP threshold (mV)": -60 + 0.5 * x,
                            LOG_TARGETS[0]: np.exp(0.05 * x)}, index=pd.Index(ids, name="cell_id"))
    targets.to_csv(prepared / "targets_raw.csv")
    author = targets.copy()
    author[LOG_TARGETS[0]] = np.log(author[LOG_TARGETS[0]])
    author.to_csv(prepared / "targets_author.csv")
    pd.DataFrame({"mouse": [f"mouse{i // 2}" for i in range(12)],
                  "rna_family": ["family_a"] * 6 + ["family_b"] * 6}, index=targets.index).to_csv(prepared / "metadata.csv")
    folds = [{"train_ids": ids[:8], "test_ids": ids[8:]}, {"train_ids": ids[8:], "test_ids": ids[:8]}]
    for protocol in ("cell_kfold", "mouse_group_kfold"):
        (splits / f"{protocol}.json").write_text(json.dumps({"scheme": protocol, "seed": 42, "folds": folds}))
    np.savez(embeddings / "genept.npz", values=expression[:, :3], cell_ids=np.array(ids),
             feature_names=np.array(["embedding0", "embedding1", "embedding2"]))
    atomic_json(embeddings / "genept.audit.json", {
        "output_sha256": file_sha256(embeddings / "genept.npz"), "frozen": True, "labels_used": False})
    channel_path = tmp_path / "channels.csv"
    pd.DataFrame({"Approved symbol": ["HCN1", "CACNA1E", "ABSENT"],
                  "Approved name": ["channel1", "channel2", "absent"]}).to_csv(channel_path, index=False)
    config = {"seed": 42, "paths": {"prepared_dir": "prepared", "splits_dir": "splits",
              "embeddings_dir": "embeddings", "results_dir": "results", "cache_dir": "cache"},
              "evaluation": {"protocols": ["cell_kfold", "mouse_group_kfold"], "minimum_gene_cells": 2,
                  "pca_components": 128, "inner_folds": 3, "ridge_alphas": [0.1, 1, 10],
                  "ion_channels_path": str(channel_path), "bootstrap_samples": 100, "family_min_cells": 2},
              "models": {}, "cluster": {"remaining_gpu_hours": None}}
    return tmp_path, config, ids, targets


def test_embedding_provenance_is_required_and_checksum_checked(benchmark):
    root, config, ids, _ = benchmark
    _, _, _, inputs = ev._features(config, root, "genept", ids[:8], ids[8:])
    audit_path = root / "embeddings/genept.audit.json"
    assert str(audit_path) in inputs
    atomic_json(audit_path, {"output_sha256": "changed"})
    with pytest.raises(ValueError, match="checksum"):
        ev._features(config, root, "genept", ids[:8], ids[8:])


def test_mean_target_inversion_and_atomic_resume(benchmark, monkeypatch):
    root, config, ids, targets = benchmark
    result = ev.evaluate_fold(config, root=root, protocol="cell_kfold", representation="genept", predictor="mean", fold=0)
    frame = pd.read_csv(result["predictions"])
    logged = frame[frame.target == LOG_TARGETS[0]]
    expected = np.exp(np.log(targets.loc[ids[:8], LOG_TARGETS[0]]).mean())
    np.testing.assert_allclose(logged.y_pred, expected)
    raw = frame[frame.target == "AP threshold (mV)"]
    np.testing.assert_allclose(raw.y_pred, targets.loc[ids[:8], "AP threshold (mV)"].mean())
    def forbidden(*args):
        raise AssertionError("completed targets must resume without fitting")
    monkeypatch.setattr(ev, "_target_transform", forbidden)
    # Supplying/updating a quota does not change a fitted model's inputs.
    config["cluster"]["remaining_gpu_hours"] = 99
    resumed = ev.evaluate_fold(config, root=root, protocol="cell_kfold", representation="genept", predictor="mean", fold=0)
    assert resumed["fingerprint"] == result["fingerprint"]
    config["seed"] = 43
    with pytest.raises(ValueError, match="context changed"):
        ev.evaluate_fold(config, root=root, protocol="cell_kfold", representation="genept", predictor="mean", fold=0)


def test_partial_run_resumes_only_unfinished_target(benchmark, monkeypatch):
    root, config, ids, targets = benchmark
    calls = []
    class Spy:
        def predict(self, train, y, query, run_id):
            calls.append((train.shape, y.copy(), query.shape))
            if len(calls) == 2:
                raise RuntimeError("synthetic interruption")
            return np.zeros(len(query)), {"synthetic": True}
    monkeypatch.setitem(sys.modules, "cell_property_prediction.models", types.SimpleNamespace(build_predictor=lambda *a, **k: Spy()))
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        ev.evaluate_fold(config, root=root, protocol="cell_kfold", representation="genept", predictor="spy", fold=0)
    result = ev.evaluate_fold(config, root=root, protocol="cell_kfold", representation="genept", predictor="spy", fold=0)
    assert len(calls) == 3  # two first-attempt calls plus only the unfinished target
    assert len(result["targets"]) == 2
    for shape, y, query_shape in calls:
        assert shape == (8, 3) and query_shape == (4, 3)
        assert abs(y.mean()) < 1e-10
        assert abs(y.std() - 1) < 1e-10


def test_features_filter_only_training_and_cap_pca(benchmark):
    root, config, ids, _ = benchmark
    train, query, audit, _ = ev._features(config, root, "pca128", ids[:8], ids[8:])
    assert "TestOnly" not in audit["genes"]
    assert "Arpp21" not in audit["genes"]
    assert train.shape == (8, 4) and query.shape == (4, 4)
    source = root / "prepared/expression.npz"
    with np.load(source) as archive:
        data = {name: archive[name] for name in archive.files}
    data["values"][8:] *= 1e6
    np.savez(source, **data)
    altered_train, _, altered_audit, _ = ev._features(config, root, "pca128", ids[:8], ids[8:])
    np.testing.assert_allclose(train, altered_train)
    assert audit["genes"] == altered_audit["genes"]
    _, _, panel_audit, _ = ev._features(config, root, "authors25", ids[:8], ids[8:])
    assert "Arpp21" in panel_audit["genes"]  # fixed panel is not subject to detection-count filtering


def test_channel_symbol_mapping_and_missing_embedding_cells(benchmark):
    root, config, ids, _ = benchmark
    train, query, audit, _ = ev._features(config, root, "ion_channels", ids[:8], ids[8:])
    assert train.shape == (8, 2) and query.shape == (4, 2)
    assert audit["genes"] == ["Hcn1", "Cacna1e"]
    with pytest.raises(ValueError, match="lacks 1 split cells"):
        ev._features(config, root, "genept", ids[:8], ["unknown"])


def test_grouped_ridge_refits_inner_preprocessing(benchmark, monkeypatch):
    root, config, ids, _ = benchmark
    original = ev._features
    calls = []
    def record(config, root, representation, train_ids, test_ids):
        calls.append((list(train_ids), list(test_ids)))
        return original(config, root, representation, train_ids, test_ids)
    monkeypatch.setattr(ev, "_features", record)
    result = ev.evaluate_fold(config, root=root, protocol="mouse_group_kfold", representation="pca128", predictor="ridge", fold=0)
    assert len(calls) == 4  # outer fit and three cached inner fits shared by both targets
    for train_ids, validation_ids in calls:
        mouse = lambda cell: int(cell.removeprefix("cell")) // 2
        assert not {mouse(cell) for cell in train_ids} & {mouse(cell) for cell in validation_ids}
    assert result["targets"][0]["audit"]["inner_scheme"] == "mouse_group_kfold"


def test_missing_labels_excluded_and_group_overlap_rejected(benchmark):
    root, config, ids, targets = benchmark
    targets.loc[ids[0], "AP threshold (mV)"] = np.nan
    targets.to_csv(root / "prepared/targets_raw.csv")
    result = ev.evaluate_fold(config, root=root, protocol="cell_kfold", representation="genept", predictor="mean", fold=0)
    assert result["targets"][0]["n_train"] == 7
    metadata = pd.read_csv(root / "prepared/metadata.csv", index_col=0)
    metadata.loc[ids[8], "mouse"] = metadata.loc[ids[0], "mouse"]
    metadata.to_csv(root / "prepared/metadata.csv")
    with pytest.raises(ValueError, match="leaks a mouse"):
        ev.evaluate_fold(config, root=root, protocol="mouse_group_kfold", representation="genept", predictor="mean", fold=0)


def test_report_full_oof_raw_metrics_and_paired_mouse_bootstrap(benchmark):
    root, config, ids, targets = benchmark
    for predictor in ("mean", "ridge"):
        for fold in range(2):
            ev.evaluate_fold(config, root=root, protocol="cell_kfold", representation="genept", predictor=predictor, fold=fold)
    # A second protocol with one fold remains partial and receives no rank.
    ev.evaluate_fold(config, root=root, protocol="mouse_group_kfold", representation="genept", predictor="mean", fold=0)
    manifest = report(config, root)
    assert manifest["n_methods_complete_oof"] == 2
    assert manifest["bootstrap"]["unit"] == "mouse"
    summary = pd.read_csv(root / "results/reports/summary.csv")
    complete = summary[summary.complete_oof]
    assert complete["rank"].notna().all()
    assert summary.loc[~summary.complete_oof, "rank"].isna().all()
    differences = pd.read_csv(root / "results/reports/paired_differences.csv")
    assert "__macro__" in set(differences.target)
    assert (differences.n_mice == 6).all()
    assert (differences.bootstrap_valid > 0).all()
    families = pd.read_csv(root / "results/reports/within_rna_family.csv")
    assert set(families.rna_family) == {"family_a", "family_b"}
    oof = pd.read_csv(root / "results/reports/oof_predictions.csv")
    selected = oof[(oof.predictor == "mean") & (oof.protocol == "cell_kfold") & (oof.target == LOG_TARGETS[0])]
    target_metrics = pd.read_csv(root / "results/reports/target_metrics.csv")
    row = target_metrics[(target_metrics.predictor == "mean") & (target_metrics.protocol == "cell_kfold") &
                         (target_metrics.target == LOG_TARGETS[0])].iloc[0]
    assert row.r2 == pytest.approx(metrics(selected.y_true, selected.y_pred)["r2"])
    source = root / "results/cell_kfold/genept/mean/fold_00/predictions.csv"
    damaged = pd.read_csv(source)
    damaged.loc[0, "y_pred"] += 1
    damaged.to_csv(source, index=False)
    with pytest.raises(ValueError, match="missing or altered"):
        report(config, root)


def test_pilot_target_subset_is_explicit(benchmark):
    root, config, ids, targets = benchmark
    config["evaluation"]["targets"] = [LOG_TARGETS[0]]
    config["paths"]["results_dir"] = "results/pilot"
    result = ev.evaluate_fold(config, root=root, protocol="cell_kfold", representation="genept", predictor="mean", fold=0)
    assert result["target_names"] == [LOG_TARGETS[0]]
    assert result["pilot_target_subset"] is True
    assert len(result["targets"]) == 1


def test_constant_and_missing_metrics_are_not_forced_to_zero():
    result = metrics([1, 1, np.nan], [0, 1, 100])
    assert result["n"] == 2 and result["mae"] == 0.5
    assert np.isnan(result["r2"]) and np.isnan(result["pearson"])
