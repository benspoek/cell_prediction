"""Authors' transformed joint R2 on the benchmark's actual saved folds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import LOG_TARGETS


def authors_joint_scores(predictions: pd.DataFrame, targets_raw: pd.DataFrame,
                         splits: list[dict], target_names: list[str]) -> tuple[dict, list[dict]]:
    """Use full-cohort population scales and equally weighted complete fold scores.

    Invalid log domains or incomplete data produce an explicit missing secondary
    score; they never change predictions, drop targets, or clip model outputs.
    """
    names = list(target_names)
    rows = [{"fold": i, "authors_joint_r2": np.nan, "sse": np.nan, "sst": np.nan,
             "n_predictions": 0, "n_targets": len(names), "status": "incomplete_oof",
             "reason": "Missing fold predictions"} for i in range(len(splits))]

    def finish(status, reason=""):
        valid = [row["authors_joint_r2"] for row in rows if row["status"] == "complete"]
        value = float(np.mean(valid)) if status == "complete" and valid else np.nan
        return {"authors_joint_r2": value, "authors_joint_r2_status": status,
                "authors_joint_r2_reason": reason, "authors_joint_r2_n_folds": len(valid)}, rows

    def invalid(reason):
        for row in rows:
            row.update(status="invalid", reason=reason)
        return finish("invalid", reason)

    if not splits or not names or len(set(names)) != len(names):
        return invalid("Expected nonempty splits and unique target names")
    if targets_raw.index.has_duplicates or targets_raw.columns.has_duplicates:
        return invalid("Duplicate cohort cell IDs or target names")
    raw = targets_raw.copy()
    raw.index = raw.index.astype(str)
    if raw.index.has_duplicates or not set(names).issubset(raw.columns):
        return invalid("Unknown targets or duplicate string cell IDs")
    truth = raw.loc[:, names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(truth).all():
        return invalid("Non-finite cohort target values")
    logged = [i for i, name in enumerate(names) if name in LOG_TARGETS]
    if logged and (truth[:, logged] <= 0).any():
        return invalid("Nonpositive cohort values for a logarithmic target")
    truth[:, logged] = np.log(truth[:, logged])
    with np.errstate(over="ignore", invalid="ignore"):
        scale = np.std(truth, axis=0, ddof=0)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        return invalid("Zero or non-finite full-cohort target standard deviation")
    transformed = pd.DataFrame(truth, index=raw.index, columns=names)
    required = {"fold", "cell_id", "target", "y_true", "y_pred"}
    if not required.issubset(predictions.columns):
        return invalid("Missing prediction columns")
    frame = predictions.copy()
    frame["cell_id"] = frame.cell_id.astype(str)
    numeric_folds = pd.to_numeric(frame.fold, errors="coerce")
    if (not np.isfinite(numeric_folds).all() or (numeric_folds != np.floor(numeric_folds)).any()
            or not numeric_folds.isin(range(len(splits))).all()):
        return invalid("Unknown or invalid prediction fold")
    frame["fold"] = numeric_folds.astype(int)
    if (not set(frame.cell_id).issubset(raw.index) or not set(frame.target).issubset(names)
            or frame.duplicated(["cell_id", "target"]).any()):
        return invalid("Unknown or duplicate prediction cell-target pairs")
    all_test_ids = []
    for i, split in enumerate(splits):
        try:
            train_ids = list(map(str, split["train_ids"]))
            test_ids = list(map(str, split["test_ids"]))
        except (KeyError, TypeError):
            return invalid(f"Invalid split {i}")
        if (not train_ids or not test_ids or len(set(train_ids)) != len(train_ids)
                or len(set(test_ids)) != len(test_ids) or set(train_ids) & set(test_ids)
                or not set(train_ids + test_ids).issubset(raw.index)):
            return invalid(f"Invalid training or test cell IDs in split {i}")
        all_test_ids.extend(test_ids)
        selected = frame[frame.fold == i]
        row = rows[i]
        row["n_predictions"] = len(selected)
        if not set(selected.cell_id).issubset(test_ids):
            row.update(status="invalid", reason="Predictions outside the declared test fold")
            continue
        if (len(selected) != len(test_ids) * len(names)
                or set(selected.cell_id) != set(test_ids) or set(selected.target) != set(names)):
            row["reason"] = "Incomplete test-cell or target coverage"
            continue
        estimated = selected.pivot(index="cell_id", columns="target", values="y_pred").reindex(
            index=test_ids, columns=names).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        observed = selected.pivot(index="cell_id", columns="target", values="y_true").reindex(
            index=test_ids, columns=names).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if (not np.isfinite(estimated).all() or not np.isfinite(observed).all()
                or not np.allclose(observed, raw.loc[test_ids, names].to_numpy(float))):
            row.update(status="invalid", reason="Non-finite predictions or mismatched observations")
            continue
        if logged and (estimated[:, logged] <= 0).any():
            row.update(status="invalid", reason="Nonpositive prediction for a logarithmic target")
            continue
        estimated[:, logged] = np.log(estimated[:, logged])
        actual = transformed.loc[test_ids, names].to_numpy()
        training_mean = transformed.loc[train_ids, names].mean(axis=0).to_numpy()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            sse = float(np.square((actual - estimated) / scale).sum())
            sst = float(np.square((actual - training_mean) / scale).sum())
            score = 1 - sse / sst if sst > 0 else np.nan
        row.update(sse=sse, sst=sst)
        if not np.isfinite(score) or not np.isfinite(sse) or not np.isfinite(sst) or sst <= 0:
            row.update(status="invalid", reason="Zero or non-finite fold denominator/error")
            continue
        row.update(authors_joint_r2=float(score), status="complete", reason="")
    invalid_rows = [row for row in rows if row["status"] == "invalid"]
    if invalid_rows:
        return finish("invalid", "; ".join(f"fold {row['fold']}: {row['reason']}" for row in invalid_rows))
    if (len(all_test_ids) != len(set(all_test_ids)) or set(all_test_ids) != set(raw.index)
            or any(row["status"] != "complete" for row in rows)):
        return finish("incomplete_oof", "All cohort cells, targets and expected folds are required")
    return finish("complete")
