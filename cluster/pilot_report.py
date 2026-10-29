#!/usr/bin/env python3
"""Compare saved pilots on cell-CV fold zero; never train or change production."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

# This is a small CPU report, also runnable on the shared GPU login node.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np
import pandas as pd

from cell_property_prediction.author_metrics import authors_joint_scores
from cell_property_prediction.config import atomic_json, compute_config, file_sha256, load_config, resolve
from cell_property_prediction.data import TARGET_COLUMNS
from cell_property_prediction.evaluation import _atomic_csv, _canonical_hash, _load_table, _mouse_column
from cell_property_prediction.jobs import compute_fingerprint
from cell_property_prediction.reporting import metrics

REPRESENTATIONS = ("pca128", "authors25", "ion_channels", "genept", "uce", "nicheformer",
                   "hvg", "hvg_pca128", "hvg_genept", "hvg_uce", "hvg_nicheformer")
BASELINES = {("tabpfn", name) for name in ("pca128", "authors25", "ion_channels")}
PAIRS = {(model, name) for model in ("tabpfn", "limix", "kumo") for name in REPRESENTATIONS}


def read_fold(folder, identity, expected_config, targets, metadata, splits, common_sources):
    """Reject altered, mismatched or incomplete saved comparisons before scoring."""
    source = folder / "manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    expected = dict(protocol="cell_kfold", fold=0, predictor=identity[0], representation=identity[1])
    if manifest.get("status") != "complete" or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Wrong or incomplete pilot identity: {source}")
    if manifest.get("config_sha256") != _canonical_hash(compute_config(expected_config)):
        raise ValueError(f"Pilot configuration differs: {source}")
    names = list(TARGET_COLUMNS)
    if manifest.get("target_names") != names:
        raise ValueError(f"Expected all seventeen targets: {source}")
    split = splits[0]
    if (manifest.get("n_train") != len(split["train_ids"]) or
            manifest.get("n_test") != len(split["test_ids"])):
        raise ValueError(f"Pilot split sizes differ: {source}")
    inputs = {str(Path(path).resolve()): digest for path, digest in manifest.get("input_sha256", {}).items()}
    if any(inputs.get(str(path.resolve())) != digest for path, digest in common_sources.items()):
        raise ValueError(f"Pilot target/metadata/split inputs differ: {source}")
    prediction = folder / "predictions.csv"
    if file_sha256(prediction) != manifest.get("prediction_sha256"):
        raise ValueError(f"Prediction checksum differs: {prediction}")
    frame = pd.read_csv(prediction, dtype={"cell_id": str, "mouse": str})
    required = set(expected) | {"cell_id", "mouse", "target", "y_true", "y_pred"}
    if not required.issubset(frame.columns) or any(not (frame[k] == v).all() for k, v in expected.items()):
        raise ValueError(f"Prediction identity differs: {prediction}")
    if (frame.duplicated(["cell_id", "target"]).any() or
            len(frame) != len(split["test_ids"]) * len(names) or
            set(frame.target) != set(names) or set(frame.cell_id) != set(map(str, split["test_ids"]))):
        raise ValueError(f"Incomplete or duplicate fold coverage: {prediction}")
    if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy(float)).all():
        raise ValueError(f"Non-finite predictions or labels: {prediction}")
    truth = np.array([targets.at[cell, target] for cell, target in zip(frame.cell_id, frame.target)], float)
    mice = metadata.loc[frame.cell_id, _mouse_column(metadata)].astype(str).to_numpy()
    if not np.allclose(truth, frame.y_true.to_numpy(float)) or not np.array_equal(mice, frame.mouse.to_numpy(str)):
        raise ValueError(f"Pilot labels or mice disagree: {prediction}")
    return frame, {str(source): file_sha256(source), str(prediction): file_sha256(prediction)}


def score(frame, targets, splits):
    """Reuse the exact production metric functions; expose only fold zero's joint score."""
    rows = [{"target": name, **metrics(selected.y_true, selected.y_pred)}
            for name in TARGET_COLUMNS for selected in [frame[frame.target == name]]]
    _, folds = authors_joint_scores(frame, targets, splits, list(TARGET_COLUMNS))
    joint = folds[0]
    if joint["status"] != "complete" or not all(np.isfinite(row["r2"]) for row in rows):
        raise ValueError(f"Cannot score all targets on pilot fold: {joint['reason']}")
    return {"macro_r2_raw": float(np.mean([row["r2"] for row in rows])),
            "authors_joint_r2_fold": joint["authors_joint_r2"], "complete_oof": False,
            "protocol": "cell_kfold", "fold": 0, "n_folds": 1, "n_targets": len(rows),
            "n_test_cells": frame.cell_id.nunique(), "n_test_mice": frame.mouse.nunique()}, rows


def run(root, config_path):
    root = Path(root).resolve()
    config = load_config(resolve(root, config_path))
    results = resolve(root, config["paths"]["results_dir"])
    pilot = root / "results/pending_workflow/pilot_with_hvg_encoders"
    timing_path = pilot / "timings.json"
    timings = json.loads(timing_path.read_text(encoding="utf-8"))
    records = timings.get("runs", [])
    found = {(row["predictor"], row["representation"]) for row in records}
    if (timings.get("pilot_targets") != 17 or found != PAIRS - BASELINES or len(records) != 30 or
            any(row.get("targets") != 17 or row.get("device") not in {"cuda", "cuda:0"} for row in records)):
        raise ValueError("Expected all thirty seventeen-target CUDA pilot measurements")
    if timings.get("fingerprint") != compute_fingerprint(config, root):
        raise ValueError("Pilot input/code/configuration provenance changed")
    prepared = resolve(root, config["paths"]["prepared_dir"])
    target_path, metadata_path = prepared / "targets_raw.csv", prepared / "metadata.csv"
    split_path = resolve(root, config["paths"]["splits_dir"]) / "cell_kfold.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))["folds"]
    targets, metadata = _load_table(target_path), _load_table(metadata_path)
    common = {path: file_sha256(path) for path in (target_path, metadata_path, split_path)}
    sources = {str(path): digest for path, digest in common.items()}
    sources[str(timing_path)] = file_sha256(timing_path)
    pilot_config = copy.deepcopy(config)
    pilot_config["evaluation"]["targets"] = list(TARGET_COLUMNS)
    pilot_config["paths"]["results_dir"] = str(pilot)
    times = {(row["predictor"], row["representation"]): row["elapsed_seconds"] for row in records}
    summaries, per_target = [], []
    for pair in sorted(PAIRS):
        baseline = pair in BASELINES
        folder = (results if baseline else pilot) / "cell_kfold" / pair[1] / pair[0] / "fold_00"
        frame, checksums = read_fold(folder, pair, config if baseline else pilot_config,
                                    targets, metadata, splits, common)
        summary, rows = score(frame, targets, splits)
        identity = {"predictor": pair[0], "representation": pair[1]}
        summaries.append({**identity, **summary, "pilot_seconds": times.get(pair, np.nan),
                          "source": "original_benchmark_fold_0" if baseline else "pilot_fold_0"})
        per_target.extend({**identity, **row} for row in rows)
        sources.update(checksums)
    summary = pd.DataFrame(summaries).sort_values("macro_r2_raw", ascending=False)
    destination = root / "diagnostics/pilot_report"
    _atomic_csv(destination / "summary.csv", summary)
    _atomic_csv(destination / "per_target.csv", pd.DataFrame(per_target))
    _atomic_csv(destination / "macro_r2_matrix.csv", summary.pivot(
        index="representation", columns="predictor", values="macro_r2_raw").reset_index())
    atomic_json(destination / "manifest.json", {
        "kind": "exploratory_single_fold_comparison", "complete_oof": False,
        "n_combinations": len(summary), "protocol": "cell_kfold", "fold": 0,
        "warning": "One saved test fold; no full-CV ranking, significance claim, or mouse-held-out result",
        "primary": "Mean of seventeen raw-scale per-target R2 values on the same saved test cells",
        "secondary": "Authors' joint scoring definition for fold zero; full-cohort scales and saved training means",
        "sources_sha256": sources,
        "outputs_sha256": {name: file_sha256(destination / name) for name in
                           ("summary.csv", "per_target.csv", "macro_r2_matrix.csv")}})
    print("EXPLORATORY: cell-CV fold 0 only; all methods use identical test cells and 17 targets.")
    print(summary[["representation", "predictor", "macro_r2_raw", "authors_joint_r2_fold"]].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Reports: {destination}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=os.environ.get("CPP_ROOT", "."))
    parser.add_argument("--config", default=os.environ.get("CPP_CONFIG", "configs/cluster.local.yaml"))
    args = parser.parse_args()
    run(args.root, args.config)
