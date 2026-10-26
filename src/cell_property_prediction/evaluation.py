"""Leakage-aware outer-fold evaluation with atomic target-level resumability."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from .config import atomic_json, compute_config, file_sha256, resolve
from .hvg import HVG_REPRESENTATIONS

FORMAT_VERSION = 1
BASELINE_REPRESENTATIONS = {"pca128", "ion_channels", "authors25"}
HVG_EMBEDDING_REPRESENTATIONS = {"hvg_genept", "hvg_uce", "hvg_nicheformer"}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _load_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate cell IDs in {path}")
    return frame


def _path(config: dict, root: Path, name: str, default: str) -> Path:
    return resolve(root, config.get("paths", {}).get(name, default))


def _mouse_column(metadata: pd.DataFrame) -> str:
    for name in ("mouse", "Mouse", "mouse_id", "donor_id", "specimen_donor_id", "donor"):
        if name in metadata:
            if metadata[name].isna().any():
                raise ValueError(f"Missing grouping values in metadata.{name}")
            return name
    raise ValueError("metadata.csv must contain a mouse or Mouse column")


def _model_file_inputs(root: Path, model_config: dict) -> dict[str, str]:
    """Hash explicitly configured local model assets, including directory inputs."""
    inputs: dict[str, str] = {}
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, (str, Path)):
            candidate = resolve(root, value)
            if candidate.is_file():
                inputs[str(candidate.resolve())] = file_sha256(candidate)
            elif candidate.is_dir():
                for item in sorted(candidate.rglob("*")):
                    if item.is_file() and ".git" not in item.parts and "__pycache__" not in item.parts:
                        inputs[str(item.resolve())] = file_sha256(item)
    visit(model_config)
    return inputs


def _read_ion_genes(path: Path, gene_names: np.ndarray) -> tuple[set[str], dict]:
    """Read HGNC group exports and simple symbol lists without positional guesses."""
    frame = pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")
    names = {str(gene).casefold(): str(gene) for gene in gene_names}
    symbol_columns = [c for c in frame if str(c).casefold().replace("_", " ").strip()
                      in {"symbol", "gene", "gene symbol", "approved symbol", "hgnc symbol", "gene name"}]
    values = frame[symbol_columns].to_numpy().ravel() if symbol_columns else frame.to_numpy().ravel()
    requested = {str(value).strip() for value in values if str(value).strip()}
    requested.update(str(c).strip() for c in frame.columns if str(c).casefold() in names)
    matched = {names[symbol.casefold()] for symbol in requested if symbol.casefold() in names}
    if not matched:
        raise ValueError(f"No ion-channel panel symbols match expression genes: {path}")
    return matched, {"source": str(path), "sha256": file_sha256(path),
                     "requested_symbols": sorted(requested), "matched_symbols": sorted(matched),
                     "matching": "case-insensitive symbols; no inferred orthology"}


def _features(config: dict, root: Path, representation: str, train_ids: list[str],
              test_ids: list[str], *, protocol: str | None = None,
              fold: int | None = None) -> tuple[np.ndarray, np.ndarray, dict, dict[str, str]]:
    if representation in HVG_EMBEDDING_REPRESENTATIONS:
        if protocol is None or fold is None:
            raise ValueError("HVG encoder features require an outer protocol and fold")
        from .hvg_embeddings import load_hvg_features
        return load_hvg_features(config, root, representation, protocol, fold, train_ids, test_ids)
    prepared = _path(config, root, "prepared_dir", "data/processed")
    if representation in HVG_REPRESENTATIONS:
        from .hvg import build_features
        return build_features(config, prepared, representation, train_ids, test_ids)
    if representation not in BASELINE_REPRESENTATIONS:
        source = _path(config, root, "embeddings_dir", "embeddings") / f"{representation}.npz"
        with np.load(source, allow_pickle=False) as archive:
            values = archive["values"].astype(np.float32)
            cell_ids = archive["cell_ids"].astype(str)
            feature_names = archive["feature_names"].astype(str).tolist() if "feature_names" in archive else None
        if values.ndim != 2 or len(values) != len(cell_ids) or len(set(cell_ids)) != len(cell_ids):
            raise ValueError(f"Invalid cell embedding shape or duplicate IDs: {source}")
        index = {cell: i for i, cell in enumerate(cell_ids)}
        missing = set(train_ids + test_ids) - index.keys()
        if missing:
            raise ValueError(f"Embedding {representation} lacks {len(missing)} split cells")
        train = values[[index[cell] for cell in train_ids]]
        test = values[[index[cell] for cell in test_ids]]
        if not np.isfinite(train).all() or not np.isfinite(test).all():
            raise ValueError(f"Non-finite features in {source}")
        provenance = source.with_suffix(".audit.json")
        inputs = {str(source): file_sha256(source)}
        if not provenance.is_file():
            raise FileNotFoundError(f"Embedding audit is required: {provenance}")
        audit = json.loads(provenance.read_text(encoding="utf-8"))
        if audit.get("output_sha256") != inputs[str(source)]:
            raise ValueError(f"Embedding checksum differs from its audit: {source}")
        inputs[str(provenance)] = file_sha256(provenance)
        return train, test, {"kind": "frozen_embedding", "n_features": values.shape[1],
                             "feature_names": feature_names}, inputs

    source = prepared / "expression.npz"
    with np.load(source, allow_pickle=False) as archive:
        values = archive["values"].astype(np.float32)
        cell_ids = archive["cell_ids"].astype(str)
        genes = archive["gene_names"].astype(str)
    if values.shape != (len(cell_ids), len(genes)) or len(set(cell_ids)) != len(cell_ids):
        raise ValueError("expression.npz has invalid dimensions or duplicate cells")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Author log-expression must contain finite nonnegative values")
    index = {cell: i for i, cell in enumerate(cell_ids)}
    train_raw = values[[index[cell] for cell in train_ids]]
    test_raw = values[[index[cell] for cell in test_ids]]
    evaluation = config.get("evaluation", {})
    minimum = int(evaluation.get("minimum_gene_cells", 10))
    keep = (train_raw > 0).sum(axis=0) >= minimum
    inputs = {str(source): file_sha256(source)}
    panel_audit = None
    if representation == "ion_channels":
        panel = resolve(root, evaluation.get("ion_channels_path", "data/raw/ion-channel-genes-group-177.csv"))
        panel_genes, panel_audit = _read_ion_genes(panel, genes)
        keep &= np.isin(genes, list(panel_genes))
        inputs[str(panel)] = file_sha256(panel)
    elif representation == "authors25":
        from .data import AUTHORS25
        keep = np.isin(np.char.lower(genes), [str(gene).lower() for gene in AUTHORS25])
    varying = np.ptp(train_raw, axis=0) > 0
    keep &= varying
    selected = genes[keep]
    if not len(selected):
        raise ValueError(f"No {representation} genes pass training-fold expression and variance filters")
    scaler = StandardScaler().fit(train_raw[:, keep])
    train, test = scaler.transform(train_raw[:, keep]), scaler.transform(test_raw[:, keep])
    audit: dict[str, Any] = {"kind": "training_fold_expression", "minimum_gene_cells": minimum,
                             "genes": selected.tolist(), "n_selected_genes": len(selected),
                             "expression_transform": "prepared author log2(1+length-corrected TPM); train-only gene scaling",
                             "panel": panel_audit}
    if representation == "pca128":
        components = min(int(evaluation.get("pca_components", 128)), train.shape[0] - 1, train.shape[1])
        if components < 1:
            raise ValueError("PCA requires at least two training cells")
        pca = PCA(n_components=components, random_state=int(config.get("seed", 42)), svd_solver="randomized")
        train = pca.fit_transform(train)
        test = pca.transform(test)
        audit.update({"pca_components": components,
                      "explained_variance_ratio": pca.explained_variance_ratio_.tolist()})
    audit["n_features"] = train.shape[1]
    return train.astype(np.float32), test.astype(np.float32), audit, inputs


def _ridge_predict(config: dict, train: np.ndarray, y: np.ndarray, query: np.ndarray,
                   groups: np.ndarray, protocol: str, inner_feature_builder=None) -> tuple[np.ndarray, dict]:
    evaluation = config.get("evaluation", {})
    alphas = [float(x) for x in evaluation.get("ridge_alphas", [0.1, 1, 10, 100, 1000])]
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise ValueError("ridge_alphas must be a non-empty list of nonnegative values")
    folds = min(int(evaluation.get("inner_folds", 3)), len(y))
    if "mouse" in protocol or "group" in protocol:
        folds = min(folds, len(np.unique(groups)))
        splits = list(GroupKFold(n_splits=folds, shuffle=True, random_state=int(config.get("seed", 42)))
                      .split(train, y, groups)) if folds >= 2 else []
    else:
        splits = list(KFold(n_splits=folds, shuffle=True, random_state=int(config.get("seed", 42)))
                      .split(train, y)) if folds >= 2 else []
    inner_data = []
    for inner_train, validation in splits:
        x_train, x_validation = (inner_feature_builder(inner_train, validation)
                                  if inner_feature_builder else (train[inner_train], train[validation]))
        scaler = StandardScaler().fit(x_train)
        inner_data.append((inner_train, validation, scaler.transform(x_train), scaler.transform(x_validation)))
    scores = []
    for alpha in alphas:
        squared_error, n = 0.0, 0
        for inner_train, validation, x_train, x_validation in inner_data:
            model = Ridge(alpha=alpha).fit(x_train, y[inner_train])
            error = model.predict(x_validation) - y[validation]
            squared_error += float(error @ error)
            n += len(validation)
        scores.append(squared_error / n if n else float("inf"))
    best = int(np.argmin(scores)) if splits else 0
    scaler = StandardScaler().fit(train)
    model = Ridge(alpha=alphas[best]).fit(scaler.transform(train), y)
    return model.predict(scaler.transform(query)), {
        "selected_alpha": alphas[best], "inner_folds": len(splits),
        "inner_scheme": "mouse_group_kfold" if "mouse" in protocol or "group" in protocol else "cell_kfold",
        "validation_mse_by_alpha": {str(alpha): (score if np.isfinite(score) else None)
                                    for alpha, score in zip(alphas, scores)}}


def _target_transform(y: np.ndarray, target: str) -> tuple[np.ndarray, bool]:
    from .data import LOG_TARGETS
    logged = target in LOG_TARGETS
    if logged and np.any(y <= 0):
        raise ValueError(f"Log target {target} contains nonpositive observed values")
    return (np.log(y) if logged else y).astype(float), logged


def _slug(target: str) -> str:
    readable = re.sub(r"[^a-zA-Z0-9._-]+", "_", target).strip("_")[:80] or "target"
    return f"{readable}_{hashlib.sha256(target.encode()).hexdigest()[:8]}"


def evaluate_fold(config: dict, *, root: Path, protocol: str, representation: str,
                  predictor: str, fold: int, device: str = "cuda") -> dict:
    """Evaluate one outer fold, atomically committing each completed target.

    Unobserved targets are excluded from training but their test cells may still
    receive predictions. Only finite observed pairs contribute to reporting.
    """
    root = Path(root).resolve()
    if predictor == "sparse_rrr":
        raise ValueError("Use the separate historical RRR runner for the legacy replication")
    if predictor == "ridge" and representation in HVG_EMBEDDING_REPRESENTATIONS:
        raise ValueError("Ridge on HVG encoder features requires separate inner-fold selection and encoding")
    prepared = _path(config, root, "prepared_dir", "data/processed")
    split_path = _path(config, root, "splits_dir", "data/splits") / f"{protocol}.json"
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if fold < 0 or fold >= len(split_manifest["folds"]):
        raise ValueError(f"Fold {fold} is outside {protocol} split manifest")
    split = split_manifest["folds"][fold]
    train_ids, test_ids = list(map(str, split["train_ids"])), list(map(str, split["test_ids"]))
    if not train_ids or not test_ids or set(train_ids) & set(test_ids):
        raise ValueError("Outer fold requires nonempty disjoint train and test cells")
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("Duplicate cell IDs in outer fold")
    targets_path, metadata_path = prepared / "targets_raw.csv", prepared / "metadata.csv"
    targets, metadata = _load_table(targets_path), _load_table(metadata_path)
    selected_targets = config.get("evaluation", {}).get("targets")
    if selected_targets is not None:
        if not selected_targets or len(set(selected_targets)) != len(selected_targets):
            raise ValueError("evaluation.targets must be a nonempty list of unique target names")
        missing_targets = set(selected_targets) - set(targets.columns)
        if missing_targets:
            raise ValueError(f"Unknown evaluation targets: {sorted(missing_targets)}")
        targets = targets.loc[:, selected_targets]
    missing = set(train_ids + test_ids) - (set(targets.index) & set(metadata.index))
    if missing:
        raise ValueError(f"Outer split refers to {len(missing)} unknown cells")
    mouse = _mouse_column(metadata)
    if "mouse" in protocol or "group" in protocol:
        if set(metadata.loc[train_ids, mouse]) & set(metadata.loc[test_ids, mouse]):
            raise ValueError("Mouse-held-out split leaks a mouse between training and test")
    started = time.time()
    feature_context = {"protocol": protocol, "fold": fold} if representation in HVG_EMBEDDING_REPRESENTATIONS else {}
    train, query, feature_audit, inputs = _features(config, root, representation, train_ids, test_ids,
                                                  **feature_context)
    inputs.update({str(targets_path): file_sha256(targets_path), str(metadata_path): file_sha256(metadata_path),
                   str(split_path): file_sha256(split_path)})
    model_cfg = config.get("models", {}).get(predictor, {})
    model_inputs = _model_file_inputs(root, model_cfg)
    code_inputs = {str(path): file_sha256(path) for path in
                   (Path(__file__), Path(__file__).with_name("data.py"), Path(__file__).with_name("models.py"),
                    Path(__file__).with_name("config.py"))
                   if path.is_file()}
    if representation in HVG_REPRESENTATIONS:
        for name in ("hvg.py", "rrr.py"):
            path = Path(__file__).with_name(name)
            code_inputs[str(path)] = file_sha256(path)
    if representation in HVG_EMBEDDING_REPRESENTATIONS:
        for name in ("hvg_embeddings.py", "hvg.py", "rrr.py", "embeddings.py"):
            path = Path(__file__).with_name(name)
            code_inputs[str(path)] = file_sha256(path)
    fingerprint = _canonical_hash({"format_version": FORMAT_VERSION, "config": compute_config(config),
                                    "protocol": protocol, "representation": representation,
                                    "predictor": predictor, "fold": fold, "device": device,
                                    "input_sha256": inputs, "model_input_sha256": model_inputs,
                                    "code_sha256": code_inputs, "train_ids": train_ids, "test_ids": test_ids})
    destination = (_path(config, root, "results_dir", "results") / protocol / representation /
                   predictor / f"fold_{fold:02d}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise ValueError(f"Result context changed at {destination}; choose a fresh results_dir")
    predictor_object = None
    outcomes, prediction_frames = [], []
    inner_feature_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    training_groups = metadata.loc[train_ids, mouse].astype(str).to_numpy()
    for target in targets.columns:
        target_path = destination / "targets" / f"{_slug(target)}.json"
        prediction_path = target_path.with_suffix(".csv")
        if target_path.is_file():
            result = json.loads(target_path.read_text(encoding="utf-8"))
            if result.get("fingerprint") != fingerprint:
                raise ValueError(f"Stale target results at {target_path}; choose a fresh results_dir")
            if result.get("status") == "complete" and prediction_path.is_file() and \
                    result.get("prediction_sha256") == file_sha256(prediction_path):
                outcomes.append(result)
                prediction_frames.append(pd.read_csv(prediction_path, dtype={"cell_id": str, "mouse": str}))
                continue
        target_started = time.time()
        raw_train = pd.to_numeric(targets.loc[train_ids, target], errors="coerce").to_numpy(float)
        observed = np.isfinite(raw_train)
        if observed.sum() < 2:
            result = {"target": target, "fingerprint": fingerprint, "status": "insufficient_labels",
                      "n_train": int(observed.sum()), "reason": "fewer than two finite training labels"}
            atomic_json(target_path, result)
            outcomes.append(result)
            continue
        transformed, logged = _target_transform(raw_train[observed], str(target))
        y_scaler = StandardScaler().fit(transformed.reshape(-1, 1))
        y = y_scaler.transform(transformed.reshape(-1, 1)).ravel()
        if predictor == "ridge":
            observed_ids = np.asarray(train_ids)[observed]
            def inner_features(inner_train, validation):
                inner_ids, validation_ids = observed_ids[inner_train].tolist(), observed_ids[validation].tolist()
                key = (tuple(inner_ids), tuple(validation_ids))
                if key not in inner_feature_cache:
                    x_train, x_validation, _, _ = _features(config, root, representation, inner_ids, validation_ids)
                    inner_feature_cache[key] = (x_train, x_validation)
                return inner_feature_cache[key]
            prediction, audit = _ridge_predict(config, train[observed], y, query,
                                                training_groups[observed], protocol,
                                                inner_features if representation in BASELINE_REPRESENTATIONS | HVG_REPRESENTATIONS else None)
        elif predictor == "mean":
            prediction, audit = np.full(len(test_ids), y.mean()), {"kind": "training_transformed_target_mean"}
        else:
            if predictor_object is None:
                from .models import build_predictor
                predictor_object = build_predictor(predictor, model_cfg, device=device,
                    cache_dir=_path(config, root, "cache_dir", "cache"), offline=True, root=root)
            prediction, audit = predictor_object.predict(train[observed], y, query,
                run_id=f"{protocol}/{representation}/{predictor}/{fold}/{_slug(target)}/{fingerprint[:12]}")
        prediction = np.asarray(prediction, dtype=float).reshape(-1)
        if len(prediction) != len(test_ids) or not np.isfinite(prediction).all():
            raise ValueError(f"{predictor} produced invalid predictions for {target}")
        raw_prediction = y_scaler.inverse_transform(prediction[:, None]).ravel()
        if logged:
            raw_prediction = np.exp(raw_prediction)
        if not np.isfinite(raw_prediction).all():
            raise ValueError(f"Inverse target transform overflowed for {target}")
        frame = pd.DataFrame({"cell_id": test_ids,
                              "mouse": metadata.loc[test_ids, mouse].astype(str).to_numpy(),
                              "target": target, "protocol": protocol, "fold": fold,
                              "representation": representation, "predictor": predictor,
                              "y_true": pd.to_numeric(targets.loc[test_ids, target], errors="coerce").to_numpy(),
                              "y_pred": raw_prediction})
        _atomic_csv(prediction_path, frame)
        result = {"target": target, "status": "complete", "fingerprint": fingerprint,
                  "n_train": int(observed.sum()), "n_test": len(test_ids),
                  "n_test_observed": int(np.isfinite(frame.y_true).sum()),
                  "log_transform": logged, "target_scaler_mean": float(y_scaler.mean_[0]),
                  "target_scaler_scale": float(y_scaler.scale_[0]), "audit": audit,
                  "elapsed_seconds": time.time() - target_started,
                  "predictions": str(prediction_path), "prediction_sha256": file_sha256(prediction_path)}
        atomic_json(target_path, result)
        outcomes.append(result)
        prediction_frames.append(frame)
    predictions_path = destination / "predictions.csv"
    if prediction_frames:
        _atomic_csv(predictions_path, pd.concat(prediction_frames, ignore_index=True))
    manifest = {"format_version": FORMAT_VERSION, "status": "complete", "fingerprint": fingerprint,
                "config_sha256": _canonical_hash(compute_config(config)),
                "protocol": protocol, "representation": representation, "predictor": predictor, "fold": fold,
                "n_train": len(train_ids), "n_test": len(test_ids), "features": feature_audit,
                "target_names": targets.columns.tolist(), "pilot_target_subset": selected_targets is not None,
                "input_sha256": inputs, "model_input_sha256": model_inputs, "code_sha256": code_inputs,
                "targets": outcomes, "elapsed_seconds": time.time() - started,
                "predictions": str(predictions_path) if prediction_frames else None,
                "prediction_sha256": file_sha256(predictions_path) if prediction_frames else None}
    atomic_json(manifest_path, manifest)
    return manifest
