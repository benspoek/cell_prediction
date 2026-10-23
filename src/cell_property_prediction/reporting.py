"""OOF metrics and paired uncertainty with mice, rather than cells, resampled."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import atomic_json, compute_config, file_sha256
from .evaluation import _atomic_csv, _canonical_hash, _load_table, _mouse_column, _path

KEYS = ["protocol", "representation", "predictor"]


def metrics(y_true, y_pred) -> dict:
    """Return physical-unit metrics; undefined R²/correlation remain missing."""
    true, predicted = np.asarray(y_true, float), np.asarray(y_pred, float)
    keep = np.isfinite(true) & np.isfinite(predicted)
    true, predicted = true[keep], predicted[keep]
    n = len(true)
    variance = float(((true - true.mean()) ** 2).sum()) if n else 0.0
    r2 = 1 - float(((true - predicted) ** 2).sum()) / variance if n >= 2 and variance > 0 else np.nan
    pearson = (float(np.corrcoef(true, predicted)[0, 1])
               if n >= 2 and variance > 0 and np.ptp(predicted) > 0 else np.nan)
    return {"n": n, "r2": r2, "mae": float(np.abs(true - predicted).mean()) if n else np.nan,
            "pearson": pearson}


def _interval(samples: np.ndarray) -> tuple[float, float, int]:
    samples = np.asarray(samples, float)
    samples = samples[np.isfinite(samples)]
    if not len(samples):
        return np.nan, np.nan, 0
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high), len(samples)


def _bootstrap_r2(frame: pd.DataFrame, targets: list[str], mice: list[str], draws: np.ndarray) -> np.ndarray:
    """Sufficient statistics make paired cluster bootstrap independent of cell count."""
    stats = np.zeros((len(mice), len(targets), 4), dtype=float)
    mouse_index, target_index = {m: i for i, m in enumerate(mice)}, {t: i for i, t in enumerate(targets)}
    finite = frame[np.isfinite(frame.y_true) & np.isfinite(frame.y_pred)]
    for (mouse, target), group in finite.groupby(["mouse", "target"], sort=False):
        y, prediction = group.y_true.to_numpy(float), group.y_pred.to_numpy(float)
        stats[mouse_index[str(mouse)], target_index[str(target)]] = (
            len(y), y.sum(), y @ y, float(((y - prediction) ** 2).sum()))
    n = draws @ stats[:, :, 0]
    total = draws @ stats[:, :, 1]
    squares = draws @ stats[:, :, 2]
    error = draws @ stats[:, :, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = squares - total ** 2 / n
        result = 1 - error / denominator
    result[(n < 2) | (denominator <= 0)] = np.nan
    return result


def _macro_samples(samples: np.ndarray, valid_targets: np.ndarray) -> np.ndarray:
    if not valid_targets.any():
        return np.full(samples.shape[0], np.nan)
    selected = samples[:, valid_targets]
    # A replicate with a degenerate target cannot silently change the macro's target set.
    result = np.full(len(samples), np.nan)
    complete = np.isfinite(selected).all(axis=1)
    result[complete] = selected[complete].mean(axis=1)
    return result


def report(config: dict, root: Path) -> dict:
    """Report complete OOF methods and mark partial methods without ranking them.

    Bootstrap intervals describe uncertainty in the observed OOF predictions. They
    do not include the variability from retraining under new fold assignments.
    """
    root = Path(root).resolve()
    results = _path(config, root, "results_dir", "results")
    destination = _path(config, root, "reports_dir", str(results / "reports"))
    destination.mkdir(parents=True, exist_ok=True)
    prepared = _path(config, root, "prepared_dir", "data/processed")
    targets_table, metadata = _load_table(prepared / "targets_raw.csv"), _load_table(prepared / "metadata.csv")
    target_names = list(config.get("evaluation", {}).get("targets") or targets_table.columns.astype(str).tolist())
    mouse_column = _mouse_column(metadata)
    expected_cells = set(targets_table.index)
    config_hash = _canonical_hash(compute_config(config))
    frames, manifest_rows = [], []
    sources = {str(path): file_sha256(path) for path in (prepared / "targets_raw.csv", prepared / "metadata.csv")}
    expected_splits = {}
    for protocol in config.get("evaluation", {}).get("protocols", []):
        source = _path(config, root, "splits_dir", "data/splits") / f"{protocol}.json"
        expected_splits[protocol] = json.loads(source.read_text(encoding="utf-8"))["folds"]
    for path in sorted(results.glob("*/*/*/fold_*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        if manifest.get("protocol") not in expected_splits:
            continue  # Historical/pilot namespaces have their own reports.
        if manifest.get("config_sha256") != config_hash:
            raise ValueError(f"Manifest was produced under another configuration: {path}")
        prediction_path = path.with_name("predictions.csv")
        if not manifest.get("predictions"):
            continue
        if not prediction_path.is_file() or file_sha256(prediction_path) != manifest.get("prediction_sha256"):
            raise ValueError(f"Prediction CSV missing or altered: {prediction_path}")
        frame = pd.read_csv(prediction_path, dtype={"cell_id": str, "mouse": str})
        required = set(KEYS + ["cell_id", "mouse", "target", "fold", "y_true", "y_pred"])
        if not required.issubset(frame.columns):
            raise ValueError(f"Prediction CSV lacks required fields: {prediction_path}")
        if not np.isfinite(frame.y_pred.to_numpy(float)).all():
            raise ValueError(f"Prediction CSV contains non-finite model outputs: {prediction_path}")
        for key in KEYS + ["fold"]:
            if not (frame[key] == manifest[key]).all():
                raise ValueError(f"Prediction {key} disagrees with manifest: {prediction_path}")
        protocol = manifest["protocol"]
        if protocol not in expected_splits:
            raise ValueError(f"Unexpected protocol {protocol}")
        test_ids = set(map(str, expected_splits[protocol][int(manifest["fold"])]["test_ids"]))
        if not set(frame.cell_id).issubset(test_ids):
            raise ValueError(f"Predictions include cells outside the declared test fold: {prediction_path}")
        if set(frame.target) - set(target_names):
            raise ValueError(f"Unknown prediction targets: {prediction_path}")
        actual_mice = metadata.loc[frame.cell_id, mouse_column].astype(str).to_numpy()
        if not np.array_equal(actual_mice, frame.mouse.to_numpy(str)):
            raise ValueError(f"Prediction mouse assignments disagree with metadata: {prediction_path}")
        observed_true = np.array([targets_table.at[cell, target] for cell, target in zip(frame.cell_id, frame.target)], float)
        if not np.allclose(observed_true, frame.y_true.to_numpy(float), equal_nan=True):
            raise ValueError(f"Prediction labels disagree with targets_raw.csv: {prediction_path}")
        frames.append(frame)
        context = _canonical_hash({name: manifest.get(name, {}) for name in
                                  ("config_sha256", "input_sha256", "model_input_sha256", "code_sha256")})
        manifest_rows.append({**{key: manifest[key] for key in KEYS + ["fold"]}, "path": str(path), "context": context})
        sources[str(path)] = file_sha256(path)
        sources[str(prediction_path)] = file_sha256(prediction_path)
    if not frames:
        raise ValueError("No completed fold predictions found")
    predictions = pd.concat(frames, ignore_index=True)
    if predictions.duplicated(KEYS + ["cell_id", "target"]).any():
        raise ValueError("A cell-target pair occurs in multiple outer test folds")
    _atomic_csv(destination / "oof_predictions.csv", predictions)
    folds = pd.DataFrame(manifest_rows)
    for key, selected in folds.groupby(KEYS):
        if selected.context.nunique() > 1:
            raise ValueError(f"Fold input/model/source contexts differ within {key}; do not pool these runs")
    evaluation = config.get("evaluation", {})
    n_bootstrap = int(evaluation.get("bootstrap_samples", 1000))
    if n_bootstrap < 0:
        raise ValueError("bootstrap_samples cannot be negative")
    mice = sorted(metadata.loc[sorted(expected_cells), mouse_column].astype(str).unique())
    rng = np.random.default_rng(int(config.get("seed", 42)))
    draws = rng.multinomial(len(mice), np.repeat(1 / len(mice), len(mice)), size=n_bootstrap)
    target_rows, summary_rows, family_rows, bootstraps, point_r2 = [], [], [], {}, {}
    family_column = next((name for name in ("rna_family", "RNA family", "RNA_family") if name in metadata), None)
    minimum_family = int(evaluation.get("family_min_cells", 20))
    for key, group in predictions.groupby(KEYS, sort=True):
        identity = dict(zip(KEYS, key))
        found_folds = set(folds.loc[(folds[KEYS] == pd.Series(identity)).all(axis=1), "fold"].astype(int))
        expected_folds = set(range(len(expected_splits[key[0]])))
        coverage = {target: set(group.loc[group.target == target, "cell_id"]) == expected_cells for target in target_names}
        complete = found_folds == expected_folds and all(coverage.values())
        values = []
        rows_for_method = []
        for target in target_names:
            selected = group[group.target == target]
            row = {**identity, "target": target, **metrics(selected.y_true, selected.y_pred),
                   "complete_oof": complete, "n_folds": len(found_folds)}
            target_rows.append(row)
            rows_for_method.append(row)
            values.append(row["r2"])
        values = np.asarray(values, float)
        point_r2[key] = values
        valid = np.isfinite(values)
        summary = {**identity, "complete_oof": complete, "n_folds": len(found_folds),
                   "expected_folds": len(expected_folds), "n_targets": len(target_names),
                   "n_targets_scored": int(valid.sum()), "macro_r2": float(values[valid].mean()) if valid.any() else np.nan,
                   "n_mice": group.mouse.nunique()}
        if complete and n_bootstrap:
            samples = _bootstrap_r2(group, target_names, mice, draws)
            bootstraps[key] = samples
            low, high, usable = _interval(_macro_samples(samples, valid))
            summary.update({"macro_r2_ci_low": low, "macro_r2_ci_high": high, "bootstrap_valid": usable})
            for index, row in enumerate(rows_for_method):
                low, high, usable = _interval(samples[:, index])
                row.update({"r2_ci_low": low, "r2_ci_high": high, "bootstrap_valid": usable})
        summary_rows.append(summary)
        if family_column:
            with_family = group.assign(rna_family=metadata.loc[group.cell_id, family_column].to_numpy())
            for (family, target), selected in with_family.groupby(["rna_family", "target"]):
                computed = metrics(selected.y_true, selected.y_pred)
                if computed["n"] >= minimum_family:
                    family_rows.append({**identity, "rna_family": family, "target": target,
                                        "complete_oof": complete, **computed})
    target_metrics = pd.DataFrame(target_rows)
    summary = pd.DataFrame(summary_rows)
    summary["rank"] = np.nan
    for protocol, selected in summary[summary.complete_oof].groupby("protocol"):
        summary.loc[selected.index, "rank"] = selected.macro_r2.rank(method="min", ascending=False)
    summary = summary.sort_values(["protocol", "complete_oof", "macro_r2"], ascending=[True, False, False])
    _atomic_csv(destination / "target_metrics.csv", target_metrics)
    _atomic_csv(destination / "summary.csv", summary)
    _atomic_csv(destination / "within_rna_family.csv", pd.DataFrame(family_rows,
        columns=KEYS + ["rna_family", "target", "complete_oof", "n", "r2", "mae", "pearson"]))
    differences = []
    for left, right in itertools.combinations(sorted(bootstraps), 2):
        if left[0] != right[0]:
            continue
        valid = np.isfinite(point_r2[left]) & np.isfinite(point_r2[right])
        for index in np.flatnonzero(valid):
            low, high, usable = _interval(bootstraps[left][:, index] - bootstraps[right][:, index])
            differences.append({"protocol": left[0], "method_a": "/".join(left[1:]), "method_b": "/".join(right[1:]),
                "target": target_names[index], "r2_difference_a_minus_b": point_r2[left][index] - point_r2[right][index],
                "ci_low": low, "ci_high": high, "bootstrap_valid": usable, "n_mice": len(mice)})
        if valid.any():
            delta = _macro_samples(bootstraps[left], valid) - _macro_samples(bootstraps[right], valid)
            low, high, usable = _interval(delta)
            differences.append({"protocol": left[0], "method_a": "/".join(left[1:]), "method_b": "/".join(right[1:]),
                "target": "__macro__", "r2_difference_a_minus_b": float((point_r2[left] - point_r2[right])[valid].mean()),
                "ci_low": low, "ci_high": high, "bootstrap_valid": usable, "n_mice": len(mice)})
    _atomic_csv(destination / "paired_differences.csv", pd.DataFrame(differences, columns=["protocol", "method_a", "method_b",
        "target", "r2_difference_a_minus_b", "ci_low", "ci_high", "bootstrap_valid", "n_mice"]))
    output_names = ["oof_predictions.csv", "target_metrics.csv", "summary.csv", "within_rna_family.csv", "paired_differences.csv"]
    result = {"status": "complete", "config_sha256": config_hash, "n_fold_manifests": len(manifest_rows),
              "n_prediction_rows": len(predictions), "n_methods_complete_oof": int(summary.complete_oof.sum()),
              "primary_metric": "mean of per-target OOF R2 in raw physical units",
              "missing_label_policy": "finite observed pairs; undefined metrics remain missing",
              "bootstrap": {"unit": "mouse", "samples": n_bootstrap, "seed": int(config.get("seed", 42)),
                            "paired": True, "interval": "percentile 95%; fixed OOF predictions; no retraining"},
              "historical_paper_score": "Reserved for the separate historical RRR reproduction; no proxy substituted",
              "source_sha256": sources,
              "outputs": {name: {"path": str(destination / name), "sha256": file_sha256(destination / name)}
                          for name in output_names}}
    atomic_json(destination / "manifest.json", result)
    return result
