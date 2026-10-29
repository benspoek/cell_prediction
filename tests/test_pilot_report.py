"""Single-fold comparisons must not mix splits, labels or partial pilot targets."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cell_property_prediction.config import compute_config, file_sha256
from cell_property_prediction.data import LOG_TARGETS, TARGET_COLUMNS
from cell_property_prediction.evaluation import _canonical_hash

SPEC = importlib.util.spec_from_file_location("pilot_report_test", Path(__file__).resolve().parents[1] /
                                            "cluster/pilot_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    prepared = root / "data/processed"
    prepared.mkdir(parents=True)
    ids = [f"cell{i}" for i in range(6)]
    truth = pd.DataFrame({name: np.arange(1., 7.) * (i + 1) for i, name in enumerate(TARGET_COLUMNS)},
                         index=pd.Index(ids, name="cell_id"))
    metadata = pd.DataFrame({"mouse": ["a", "a", "b", "c", "d", "e"]}, index=truth.index)
    truth.to_csv(prepared / "targets_raw.csv")
    metadata.to_csv(prepared / "metadata.csv")
    splits = [{"train_ids": ids[3:], "test_ids": ids[:3]},
              {"train_ids": ids[:3], "test_ids": ids[3:]}]
    split_path = root / "data/splits/cell_kfold.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps({"folds": splits}))
    config = {"schema_version": 1, "paths": {"results_dir": "results", "prepared_dir": "data/processed",
              "splits_dir": "data/splits"}, "evaluation": {}, "data": {}, "models": {},
              "embeddings": {}, "cluster": {}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    pilot = root / "results/pending_workflow/pilot_with_hvg_encoders"
    pilot.mkdir(parents=True)
    records = [{"predictor": model, "representation": name, "elapsed_seconds": 10., "targets": 17,
                "device": "cuda"} for model, name in sorted(report.PAIRS - report.BASELINES)]
    timing_path = pilot / "timings.json"
    timing_path.write_text(json.dumps({"runs": records, "pilot_targets": 17, "fingerprint": "frozen"}))
    monkeypatch.setattr(report, "compute_fingerprint", lambda *args: "frozen")
    pilot_config = copy.deepcopy(config)
    pilot_config["evaluation"]["targets"] = list(TARGET_COLUMNS)
    pilot_config["paths"]["results_dir"] = str(pilot)
    common = {path: file_sha256(path) for path in
              (prepared / "targets_raw.csv", prepared / "metadata.csv", split_path)}
    folders = {}
    for model, representation in sorted(report.PAIRS):
        baseline = (model, representation) in report.BASELINES
        folder = (root / "results" if baseline else pilot) / "cell_kfold" / representation / model / "fold_00"
        folder.mkdir(parents=True)
        frame = pd.DataFrame([{"cell_id": cell, "mouse": metadata.at[cell, "mouse"], "target": target,
            "protocol": "cell_kfold", "fold": 0, "predictor": model, "representation": representation,
            "y_true": truth.at[cell, target], "y_pred": truth.at[cell, target] * 1.1}
            for cell in ids[:3] for target in TARGET_COLUMNS])
        frame.to_csv(folder / "predictions.csv", index=False)
        manifest = {"status": "complete", "protocol": "cell_kfold", "fold": 0, "predictor": model,
                    "representation": representation, "n_train": 3, "n_test": 3,
                    "target_names": list(TARGET_COLUMNS),
                    "config_sha256": _canonical_hash(compute_config(config if baseline else pilot_config)),
                    "input_sha256": {str(path): digest for path, digest in common.items()},
                    "prediction_sha256": file_sha256(folder / "predictions.csv")}
        (folder / "manifest.json").write_text(json.dumps(manifest))
        folders[model, representation] = folder
    return root, folders, truth, splits, timing_path


def test_same_fold_comparison_and_joint_formula_leave_sources_unchanged(rig):
    root, folders, truth, splits, timing_path = rig
    sources = [timing_path] + [f / name for f in folders.values() for name in ("manifest.json", "predictions.csv")]
    hashes = {path: file_sha256(path) for path in sources}
    summary = report.run(root, "config.yaml")
    assert len(summary) == 33
    assert not summary.complete_oof.any()
    assert set(summary.n_folds) == {1}
    assert set(summary.n_test_cells) == {3}
    assert set(summary.n_targets) == {17}
    actual = truth.to_numpy(float)
    estimate = actual[:3] * 1.1
    for i, name in enumerate(TARGET_COLUMNS):
        if name in LOG_TARGETS:
            actual[:, i] = np.log(actual[:, i])
            estimate[:, i] = np.log(estimate[:, i])
    scale = actual.std(axis=0, ddof=0)
    expected = 1 - (((actual[:3] - estimate) / scale) ** 2).sum() / (
        ((actual[:3] - actual[3:].mean(axis=0)) / scale) ** 2).sum()
    assert summary.authors_joint_r2_fold.to_numpy() == pytest.approx(np.repeat(expected, 33))
    assert all(file_sha256(path) == digest for path, digest in hashes.items())
    target = pd.read_csv(root / "diagnostics/pilot_report/per_target.csv")
    assert len(target) == 33 * 17
    provenance = json.loads((root / "diagnostics/pilot_report/manifest.json").read_text())
    assert provenance["complete_oof"] is False


@pytest.mark.parametrize("damage", ["checksum", "coverage", "labels", "mice", "fold", "config", "split"])
def test_corrupt_or_incomparable_saved_predictions_are_rejected(rig, damage):
    root, folders, _, _, _ = rig
    folder = folders["limix", "genept"]
    path = folder / "predictions.csv"
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frame = pd.read_csv(path)
    if damage == "coverage": frame = frame.iloc[1:]
    elif damage == "labels": frame.loc[0, "y_true"] += 1
    elif damage == "mice": frame.loc[0, "mouse"] = "wrong"
    elif damage == "fold": frame["fold"] = 1
    elif damage == "config": manifest["config_sha256"] = "wrong"
    elif damage == "split": manifest["n_test"] = 99
    frame.to_csv(path, index=False)
    manifest["prediction_sha256"] = "wrong" if damage == "checksum" else file_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): report.run(root, "config.yaml")
    assert not (root / "diagnostics/pilot_report/summary.csv").exists()


@pytest.mark.parametrize("damage", ["missing_pair", "targets", "fingerprint", "duplicate_pair"])
def test_incomplete_or_stale_timing_context_is_rejected(rig, damage):
    root, _, _, _, path = rig
    value = json.loads(path.read_text())
    if damage == "missing_pair": value["runs"].pop()
    elif damage == "targets": value["runs"][0]["targets"] = 2
    elif damage == "fingerprint": value["fingerprint"] = "old"
    elif damage == "duplicate_pair": value["runs"].append(value["runs"][0])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): report.run(root, "config.yaml")
