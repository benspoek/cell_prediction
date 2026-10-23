"""Faithful historical preprocessing/splits, with explicit solver provenance."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import atomic_json, file_sha256, resolve
from .data import LOG_TARGETS, SOURCE_COMMIT, load_source_tables, prepare_cohort, read_ion_channel_genes
from .rrr import fit_rrr, gene_selection, historical_joint_r2
from .splits import make_splits


def _global_standardize(values):
    mean, scale = np.mean(values, axis=0), np.std(values, axis=0)
    if (scale <= 0).any() or not np.isfinite(scale).all():
        raise ValueError("Historical preprocessing requires nonconstant finite features")
    return (values - mean) / scale, mean, scale


def _correlations(first, second):
    result = []
    for column in range(min(first.shape[1], second.shape[1])):
        if np.std(first[:, column]) == 0 or np.std(second[:, column]) == 0:
            result.append(None)
        else:
            result.append(float(np.corrcoef(first[:, column], second[:, column])[0, 1]))
    return result


def run_historical(config: dict, root: Path, backend: str = "sklearn_compatible") -> dict:
    """Run legacy broad/ion-channel RRR, keeping its 9 never-tested cells.

    Uses all 17 original targets even when a pilot clean run requests fewer.
    The compatible solver is a reconstruction, not a claim to reproduce the
    published glmnet numerical values exactly.
    """
    from .evaluation import _atomic_csv, _canonical_hash

    started = time.time()
    root = Path(root).resolve()
    paths = config.get("paths", {})
    raw_dir = resolve(root, paths.get("raw_dir", "data/raw"))
    cohort = prepare_cohort(load_source_tables(raw_dir))
    expected = int(config.get("data", {}).get("expected_cohort_cells", 1219))
    if len(cohort.features) != expected:
        raise ValueError(f"Historical cohort has {len(cohort.features)} cells, expected {expected}")
    settings = config.get("historical", {})
    panels = settings.get("panels", ["broad", "ion_channels"])
    if not panels or not set(panels).issubset({"broad", "ion_channels"}):
        raise ValueError("Historical panels must contain broad and/or ion_channels")
    original_path = resolve(root, config.get("evaluation", {}).get(
        "original_sparse_rrr_path", "vendor/mini-atlas/code/sparseRRR.py"))
    if backend == "original" and not original_path.is_file():
        raise FileNotFoundError(f"Original sparseRRR source missing: {original_path}")
    if backend not in {"sklearn_compatible", "original"}:
        raise ValueError(f"Unknown backend: {backend}")
    inputs = {str(raw_dir / name): file_sha256(raw_dir / name) for name in (
        "m1_patchseq_exon_counts.csv.gz", "m1_patchseq_intron_counts.csv.gz",
        "gene_lengths.txt", "m1_patchseq_meta_data.csv", "m1_patchseq_ephys_features.csv")}
    code_hashes = {str(path): file_sha256(path) for path in
                   (Path(__file__), Path(__file__).with_name("rrr.py"), Path(__file__).with_name("data.py"), Path(__file__).with_name("splits.py"))}
    if backend == "original":
        code_hashes[str(original_path)] = file_sha256(original_path)
    y, ymean_global, yscale_global = _global_standardize(cohort.targets_author.to_numpy(dtype=float))
    expression = cohort.features.to_numpy(dtype=float)
    genes = cohort.features.columns.to_numpy()
    cells = cohort.features.index.tolist()
    plan = make_splits(cells, cohort.metadata, "author_historical", n_splits=10, seed=42)
    positions = {cell: position for position, cell in enumerate(cells)}
    destination = resolve(root, paths.get("results_dir", "results")) / "historical" / backend
    scopes = {}
    for panel in panels:
        if panel == "broad":
            selected = gene_selection(cohort.exon_counts, n=1000)
            penalty = .45
        else:
            channel_path = resolve(root, config.get("evaluation", {}).get(
                "ion_channels_path", "data/raw/ion-channel-genes-group-177.csv"))
            channel_genes = read_ion_channel_genes(channel_path, genes)
            inputs[str(channel_path)] = file_sha256(channel_path)
            detected = np.asarray((cohort.counts > 0).sum(axis=0)).ravel() >= 10
            selected = np.isin(genes, channel_genes) & detected
            penalty = .303
        x, _, _ = _global_standardize(expression[:, selected])
        fold_results, prediction_frames = [], []
        for number, fold in enumerate(plan.folds):
            training = np.array([positions[cell] for cell in fold.train_ids])
            testing = np.array([positions[cell] for cell in fold.test_ids])
            xmean, ymean = x[training].mean(axis=0), y[training].mean(axis=0)
            xtrain, xtest = x[training] - xmean, x[testing] - xmean
            ytrain, ytest = y[training] - ymean, y[testing] - ymean
            model = fit_rrr(xtrain, ytrain, rank=5, lambdau=penalty, alpha=1.,
                            backend=backend, original_path=original_path)
            if int(model.selected.sum()) < 5:
                fold_results.append({"fold": number, "status": "skipped_fewer_than_rank_genes",
                                     "n_selected_genes": int(model.selected.sum())})
                continue
            predicted_centered = model.predict(xtest)
            score = historical_joint_r2(ytest, predicted_centered)
            correlations = _correlations(xtest @ model.wx, ytest @ model.wy)
            fold_results.append({"fold": number, "status": "complete", "joint_r2": score,
                                 "correlations": correlations, "n_test": len(testing),
                                 "n_selected_genes": int(model.selected.sum()),
                                 "selected_genes": genes[selected][model.selected].tolist(),
                                 "converged": model.converged, "iterations": model.n_iter})
            prediction_author = (predicted_centered + ymean) * yscale_global + ymean_global
            prediction_raw = prediction_author.copy()
            for column, target in enumerate(cohort.targets_author.columns):
                if target in LOG_TARGETS:
                    prediction_raw[:, column] = np.exp(prediction_raw[:, column])
                prediction_frames.append(pd.DataFrame({
                    "cell_id": list(fold.test_ids), "fold": number, "target": target,
                    "y_true": cohort.targets_raw.iloc[testing, column].to_numpy(),
                    "y_pred": prediction_raw[:, column],
                    "y_true_author": cohort.targets_author.iloc[testing, column].to_numpy(),
                    "y_pred_author": prediction_author[:, column],
                }))
        completed = [result for result in fold_results if result["status"] == "complete"]
        if not completed:
            raise RuntimeError(f"No historical {panel} folds completed")
        predictions_path = destination / panel / "predictions.csv"
        _atomic_csv(predictions_path, pd.concat(prediction_frames, ignore_index=True))
        corr_matrix = np.array([[np.nan if value is None else value for value in result["correlations"]]
                                for result in completed])
        scopes[panel] = {
            "status": "complete" if len(completed) == 10 else "partial",
            "n_completed_folds": len(completed),
            "lambda": penalty, "alpha": 1., "rank": 5,
            "n_candidate_genes": int(selected.sum()),
            "candidate_genes": genes[selected].tolist(),
            "folds": fold_results,
            "mean_joint_r2": float(np.mean([result["joint_r2"] for result in completed])) if len(completed) == 10 else None,
            "mean_correlations": [None if not np.isfinite(value) else float(value)
                                  for value in np.nanmean(corr_matrix, axis=0)],
            "predictions": str(predictions_path), "prediction_sha256": file_sha256(predictions_path),
        }
    manifest = {
        "schema_version": 1,
        "status": "complete" if all(scope["status"] == "complete" for scope in scopes.values()) else "partial",
        "backend": backend,
        "bit_exact_original": False, "numerical_match_verified": False,
        "original_source_backend": backend == "original",
        "solver_note": "Pinned author implementation; numerical equality depends on the glmnet runtime and has not been verified" if backend == "original" else
                       "sklearn-compatible group ElasticNet objective; solver differs from glmnet; numerical reconstruction only",
        "source_commit": SOURCE_COMMIT,
        "original_notebook_blob_sha": "bc25ebb4048c7508671c8c332aea09ac2e8e937c",
        "original_sparse_rrr_blob_sha": "46e30ba6a2b67de50c14ffc6f91a7cf695e4006f",
        "source_url": f"https://github.com/berenslab/mini-atlas/blob/{SOURCE_COMMIT}/code/reduced-rank-regression.ipynb",
        "cohort_audit": cohort.cohort_audit, "n_cells": len(cells),
        "never_tested_ids": list(plan.never_tested_ids),
        "n_unique_test_cells": len(cells) - len(plan.never_tested_ids),
        "split_note": "seed42 permutation; ten floor(n/10) test blocks; remainder always trains",
        "preprocessing_note": "Historical global gene selection and global X/Y scaling occur before CV; train means are then subtracted within folds",
        "metric": "mean across folds of 1 - joint standardized SSE / training-mean-centered test SST",
        "scopes": scopes, "input_sha256": inputs, "code_sha256": code_hashes,
        "config_sha256": _canonical_hash(config), "elapsed_seconds": time.time() - started,
    }
    atomic_json(destination / "manifest.json", manifest)
    return manifest
