"""Sparse reduced-rank regression with a disclosed sklearn-compatible backend.

The alternating objective follows mini-atlas sparseRRR.py. sklearn's group
coordinate solver is not bit-exact with glmnet; historical reporting states
this explicitly. An optional original backend never silently falls back.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.model_selection import GroupKFold, KFold


@dataclass
class RRRModel:
    wx: np.ndarray
    wy: np.ndarray
    selected: np.ndarray
    backend: str
    n_iter: int
    converged: bool

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x) @ self.wx @ self.wy.T


def gene_selection(data, n: int = 1000, threshold: float = 32,
                   atleast: int = 10, decay: float = 1.5,
                   yoffset: float = .02, xoffset: float = 5) -> np.ndarray:
    """Exact zero-rate/mean-log boundary from author rnaseqTools.geneSelection.

    Run on raw exon counts from the training cells for a clean benchmark. The
    original binary search may not obtain exactly n for small/tied fixtures;
    this intentionally returns its final boundary rather than inventing ranks.
    """
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("Gene selection requires a nonempty cells-by-genes matrix")
    if n < 1 or atleast < 1 or threshold < 0:
        raise ValueError("Invalid gene-selection parameters")
    if sparse.issparse(data):
        counts = sparse.csr_matrix(data)
        if not np.isfinite(counts.data).all() or (counts.data < 0).any():
            raise ValueError("Raw exon counts must be finite and nonnegative")
        detected_counts = np.asarray((counts > threshold).sum(axis=0)).ravel()
        zero_rate = 1 - detected_counts / counts.shape[0]
        above = counts.multiply(counts > threshold).astype(float).tocsr()
        above.eliminate_zeros()
        above.data = np.log2(above.data)
        log_sums = np.asarray(above.sum(axis=0)).ravel()
    else:
        counts = np.asarray(data)
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Raw exon counts must be finite and nonnegative")
        detected = counts > threshold
        detected_counts = detected.sum(axis=0)
        zero_rate = 1 - detected_counts / counts.shape[0]
        logs = np.zeros(counts.shape, dtype=float)
        np.log2(counts, out=logs, where=detected)
        log_sums = logs.sum(axis=0)
    mean_expression = np.full(counts.shape[1], np.nan)
    np.divide(log_sums, detected_counts, out=mean_expression, where=detected_counts > 0)
    eligible = detected_counts >= atleast
    zero_rate = np.where(eligible, zero_rate, np.nan)
    mean_expression[~eligible] = np.nan
    lower, upper = 0., 10.
    selected = np.zeros(counts.shape[1], dtype=bool)
    for _ in range(100):
        selected = eligible & (zero_rate > np.exp(-decay * (mean_expression - xoffset)) + yoffset)
        size = selected.sum()
        if size == n:
            break
        if size < n:
            upper = xoffset
            xoffset = (xoffset + lower) / 2
        else:
            lower = xoffset
            xoffset = (xoffset + upper) / 2
    return selected


def _ridge_rrr(x: np.ndarray, y: np.ndarray, rank: int, penalty: float):
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    denominator = singular ** 2 + penalty * len(x)
    factors = np.divide(singular, denominator, out=np.zeros_like(singular), where=denominator > 0)
    coefficients = (vt.T * factors) @ u.T @ y
    _, _, response_vt = np.linalg.svd(x @ coefficients, full_matrices=False)
    wy = response_vt.T[:, :rank]
    wx = coefficients @ wy
    return _orient(wx, wy)


def _orient(wx, wy):
    positions = np.argmax(np.abs(wy), axis=0)
    signs = np.sign(wy[positions, np.arange(wy.shape[1])])
    signs[signs == 0] = 1
    return wx * signs, wy * signs


def _elastic_rrr(x, y, rank, lambdau, alpha, max_iter, tolerance):
    if alpha == 0:
        wx, wy = _ridge_rrr(x, y, rank, lambdau)
        return wx, wy, 1, True
    _, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    wy = vt[:rank].T
    previous_loss = np.inf
    total = np.sum(y ** 2)
    for iteration in range(max_iter):
        estimator = MultiTaskElasticNet(
            alpha=lambdau, l1_ratio=alpha, fit_intercept=False,
            max_iter=20000, tol=1e-8, selection="cyclic",
        )
        estimator.fit(x, y @ wy)
        wx = np.asarray(estimator.coef_).T
        if not np.any(wx):
            return wx, np.zeros_like(wy), iteration + 1, True
        left, _, right = np.linalg.svd(y.T @ x @ wx, full_matrices=False)
        wx, wy = _orient(wx, left @ right)
        loss = np.sum((y - x @ wx @ wy.T) ** 2) / total if total else 0.
        if iteration and abs(loss - previous_loss) < tolerance:
            return wx, wy, iteration + 1, True
        previous_loss = loss
    return wx, wy, max_iter, False


def load_original_backend(path: str | Path):
    """Explicitly load the pinned author source; missing glmnet is an error."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Original backend missing: {path}; stage the pinned mini-atlas source")
    source = path.read_bytes()
    blob_sha = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    if blob_sha != "46e30ba6a2b67de50c14ffc6f91a7cf695e4006f":
        raise ValueError("Original sparseRRR source does not match the pinned author Git blob")
    spec = importlib.util.spec_from_file_location("mini_atlas_original_sparse_rrr", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        raise ImportError("Original sparseRRR backend requires glmnet_python and author dependencies; no fallback was used") from error
    return module


def fit_rrr(x, y, *, rank: int = 5, lambdau: float = .45, alpha: float = 1.,
            relax: bool = True, backend: str = "sklearn_compatible",
            original_path: str | Path | None = None, max_iter: int = 100,
            tolerance: float = 1e-6) -> RRRModel:
    """Fit centered inputs; ridge relaxation retains the supplied lambda."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or min(x.shape + y.shape) < 1:
        raise ValueError("RRR requires aligned nonempty 2D matrices")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("RRR inputs must be finite")
    if rank < 1 or lambdau < 0 or not 0 <= alpha <= 1 or max_iter < 1:
        raise ValueError("Invalid rank, lambda, alpha or iteration count")
    effective_rank = min(rank, x.shape[1], y.shape[1], len(x))
    if backend == "original":
        if original_path is None:
            raise ValueError("original_path is required for the original backend")
        module = load_original_backend(original_path)
        method = module.relaxed_elastic_rrr if relax else module.elastic_rrr
        wx, wy = method(x, y, rank=effective_rank, lambdau=lambdau, alpha=alpha, max_iter=max_iter)
        selected = np.any(np.asarray(wx) != 0, axis=1)
        return RRRModel(np.asarray(wx), np.asarray(wy), selected, backend, max_iter, True)
    if backend != "sklearn_compatible":
        raise ValueError(f"Unknown RRR backend: {backend}")
    wx, wy, iterations, converged = _elastic_rrr(x, y, effective_rank, lambdau, alpha, max_iter, tolerance)
    selected = np.any(wx != 0, axis=1)
    if relax and alpha != 0 and selected.any():
        wr, wy = _ridge_rrr(x[:, selected], y, min(effective_rank, int(selected.sum())), lambdau)
        wx = np.zeros((x.shape[1], wr.shape[1]))
        wx[selected] = wr
    return RRRModel(wx, wy, selected, backend, iterations, converged)


def historical_joint_r2(y_centered, prediction_centered) -> float:
    """Author scalar metric: joint standardized SSE over training-centered SST."""
    denominator = float(np.sum(np.asarray(y_centered) ** 2))
    return 1 - float(np.sum((np.asarray(y_centered) - prediction_centered) ** 2)) / denominator if denominator else float("nan")


def _scaled_train_query(train, query):
    mean, scale = np.mean(train, axis=0), np.std(train, axis=0)
    scale = np.where(scale > 0, scale, 1.)
    return (train - mean) / scale, (query - mean) / scale, mean, scale


def _fold_features(exon, expression, train_positions, query_positions):
    selected = gene_selection(exon[train_positions], n=1000)
    selected &= np.ptp(expression[train_positions], axis=0) > 0
    if not selected.any():
        raise ValueError("No genes passed the author zero-rate selection in training cells")
    train, query, _, _ = _scaled_train_query(expression[train_positions][:, selected], expression[query_positions][:, selected])
    return train, query, selected


def run_rrr_fold(config: dict, root: Path, protocol: str, fold: int) -> dict:
    """Leakage-aware 1,000-gene RRR with training-only inner lambda tuning."""
    import pandas as pd
    from .config import atomic_json, compute_config, file_sha256, resolve
    from .data import LOG_TARGETS, load_prepared
    from .evaluation import _atomic_csv, _canonical_hash
    from .splits import load_splits

    started = time.time()
    root = Path(root).resolve()
    paths = config.get("paths", {})
    prepared_dir = resolve(root, paths.get("prepared_dir", "data/processed"))
    cohort = load_prepared(prepared_dir)
    split_path = resolve(root, paths.get("splits_dir", "data/splits")) / f"{protocol}.json"
    plan = load_splits(split_path, cohort.metadata)
    if plan.scheme != protocol or protocol == "author_historical":
        raise ValueError("Clean RRR requires a corrected matching split protocol")
    if not 0 <= fold < plan.n_splits:
        raise ValueError("Outer fold index is out of range")
    train_ids, test_ids = list(plan.folds[fold].train_ids), list(plan.folds[fold].test_ids)
    metadata = cohort.metadata
    mouse_column = "mouse" if "mouse" in metadata else "Mouse"
    requested_targets = config.get("evaluation", {}).get("targets")
    targets = list(cohort.targets_raw.columns) if requested_targets is None else list(requested_targets)
    if not targets or len(set(targets)) != len(targets) or not set(targets).issubset(cohort.targets_author.columns):
        raise ValueError("evaluation.targets must be a nonempty unique subset of the 17 targets")
    inputs = {str(prepared_dir / name): file_sha256(prepared_dir / name) for name in
              ("expression.npz", "counts.h5ad", "targets_raw.csv", "targets_author.csv", "metadata.csv", "manifest.json")}
    inputs[str(split_path)] = file_sha256(split_path)
    code_hash = {str(path): file_sha256(path) for path in (
        Path(__file__), Path(__file__).with_name("data.py"),
        Path(__file__).with_name("splits.py"),
        Path(__file__).with_name("config.py"),
    )}
    context = {"config": compute_config(config), "protocol": protocol, "fold": fold,
               "input_sha256": inputs, "code_sha256": code_hash, "backend": "sklearn_compatible"}
    fingerprint = _canonical_hash(context)
    destination = resolve(root, paths.get("results_dir", "results")) / protocol / "raw1000" / "sparse_rrr" / f"fold_{fold:02d}"
    manifest_path, predictions_path = destination / "manifest.json", destination / "predictions.csv"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise ValueError("Sparse-RRR result context changed; choose a fresh results_dir")
        if previous.get("status") == "complete" and predictions_path.is_file() and previous.get("prediction_sha256") == file_sha256(predictions_path):
            return previous
    index = {cell: position for position, cell in enumerate(cohort.features.index)}
    train_positions = np.array([index[cell] for cell in train_ids])
    test_positions = np.array([index[cell] for cell in test_ids])
    expression = cohort.features.to_numpy(dtype=float)
    exon = cohort.exon_counts
    y = cohort.targets_author[targets].to_numpy(dtype=float)
    groups = metadata.loc[train_ids, mouse_column].astype(str).to_numpy()
    evaluation = config.get("evaluation", {})
    settings = config.get("models", {}).get("sparse_rrr", {})
    lambdas = [float(value) for value in settings.get("lambdas", [.1, .3, .45, .5, 1.])]
    if not lambdas or not np.isfinite(lambdas).all() or min(lambdas) <= 0:
        raise ValueError("Sparse-RRR lambda candidates must be positive and finite")
    rank = int(settings.get("rank", 5))
    seed = int(config.get("seed", 42))
    inner_count = min(int(evaluation.get("inner_folds", 3)), len(train_ids))
    if "mouse" in protocol or "group" in protocol:
        inner_count = min(inner_count, len(set(groups)))
        if inner_count < 2:
            raise ValueError("At least two training mice are required for inner CV")
        inner = list(GroupKFold(n_splits=inner_count, shuffle=True, random_state=seed).split(train_positions, groups=groups))
    else:
        if inner_count < 2:
            raise ValueError("At least two training cells are required for inner CV")
        inner = list(KFold(n_splits=inner_count, shuffle=True, random_state=seed).split(train_positions))
    scores = np.zeros((len(inner), len(lambdas)))
    for inner_fold, (training, validation) in enumerate(inner):
        a, b = train_positions[training], train_positions[validation]
        xtrain, xvalid, _ = _fold_features(exon, expression, a, b)
        ytrain, yvalid, _, _ = _scaled_train_query(y[a], y[b])
        for column, penalty in enumerate(lambdas):
            candidate = fit_rrr(xtrain, ytrain, rank=rank, lambdau=penalty)
            scores[inner_fold, column] = np.mean((yvalid - candidate.predict(xvalid)) ** 2)
    penalty = lambdas[int(np.argmin(scores.mean(axis=0)))]
    xtrain, xtest, selected = _fold_features(exon, expression, train_positions, test_positions)
    ytrain, _, ymean, yscale = _scaled_train_query(y[train_positions], y[test_positions])
    model = fit_rrr(xtrain, ytrain, rank=rank, lambdau=penalty)
    predictions_author = model.predict(xtest) * yscale + ymean
    predictions = predictions_author.copy()
    for column, target in enumerate(targets):
        if target in LOG_TARGETS:
            predictions[:, column] = np.exp(predictions[:, column])
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Sparse-RRR produced non-finite predictions")
    rows = [pd.DataFrame({
        "cell_id": test_ids, "mouse": metadata.loc[test_ids, mouse_column].astype(str).to_numpy(),
        "target": target, "protocol": protocol, "fold": fold,
        "representation": "raw1000", "predictor": "sparse_rrr",
        "y_true": cohort.targets_raw.loc[test_ids, target].to_numpy(),
        "y_pred": predictions[:, column],
    }) for column, target in enumerate(targets)]
    _atomic_csv(predictions_path, pd.concat(rows, ignore_index=True))
    gene_names = cohort.features.columns.to_numpy()[selected]
    manifest = {
        "format_version": 1, "status": "complete", "fingerprint": fingerprint,
        "config_sha256": _canonical_hash(compute_config(config)), "protocol": protocol,
        "representation": "raw1000", "predictor": "sparse_rrr", "fold": fold,
        "n_train": len(train_ids), "n_test": len(test_ids),
        "backend": "sklearn_compatible", "bit_exact_original": False,
        "rank_requested": rank, "rank_effective": model.wy.shape[1],
        "lambda": penalty, "inner_lambdas": lambdas,
        "converged": model.converged, "iterations": model.n_iter,
        "inner_mse": scores.tolist(), "inner_folds": inner_count,
        "features": {"gene_selection": "raw exons >32 in >=10 training cells; author zero-rate boundary",
                     "candidate_genes": gene_names.tolist(),
                     "selected_genes": gene_names[model.selected].tolist(),
                     "preprocessing": "all-gene per-cell author expression; fold-only gene/target standardization"},
        "targets": [{"target": target, "status": "complete"} for target in targets],
        "target_names": targets,
        "pilot_target_subset": requested_targets is not None,
        "input_sha256": inputs, "code_sha256": code_hash,
        "predictions": str(predictions_path), "prediction_sha256": file_sha256(predictions_path),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(manifest_path, manifest)
    return manifest
