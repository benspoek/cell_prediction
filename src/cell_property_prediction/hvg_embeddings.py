"""Training-fold native-HVG panels and immutable frozen encoder artifacts.

The selector is exactly :func:`hvg.build_features`; its scaled expression is
never passed to an encoder. Panels contain original raw exon+intron counts.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import pandas as pd
from scipy import sparse

from . import embeddings, hvg, rrr
from .config import atomic_json, file_sha256, resolve
from .embeddings import EmbeddingResult

ENCODERS = ("genept", "uce", "nicheformer")
HVG_EMBEDDING_REPRESENTATIONS = {f"hvg_{name}" for name in ENCODERS}
DIMENSIONS = {"genept": 3072, "uce": 1280, "nicheformer": 512}
_HASH_CACHE: dict[tuple, str] = {}
_HASH_LOCK = threading.RLock()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _sha(path) -> str:
    """Memoize large immutable resource hashes, invalidating on stat changes."""
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _HASH_LOCK:
        if key not in _HASH_CACHE:
            _HASH_CACHE[key] = file_sha256(path)
        return _HASH_CACHE[key]


@contextlib.contextmanager
def _encoder_hash_cache():
    # Keep the existing encoders byte-for-byte unchanged while avoiding hashing
    # their multi-GB resources twenty times in the same preparation process.
    with _HASH_LOCK:
        original = embeddings.file_sha256
        embeddings.file_sha256 = _sha
        try:
            yield
        finally:
            embeddings.file_sha256 = original


def _path(config, root, key, default):
    return resolve(Path(root).resolve(), config.get("paths", {}).get(key, default)).resolve()


def _prepared(config, root):
    return _path(config, root, "prepared_dir", "data/processed")


def _directory(config, root, protocol, fold):
    return (_path(config, root, "embeddings_dir", "embeddings") / "hvg" /
            protocol / f"fold_{fold:02d}")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sources():
    return {Path(module.__file__).name: _sha(Path(module.__file__))
            for module in (hvg, rrr)} | {Path(__file__).name: _sha(Path(__file__))}


def _selector_context(config, root, protocol):
    prepared = _prepared(config, root)
    split_path = _path(config, root, "splits_dir", "data/splits") / f"{protocol}.json"
    return {"schema_version": 1, "protocol": protocol,
            "original_counts_sha256": _sha(prepared / "counts.h5ad"),
            "expression_sha256": _sha(prepared / "expression.npz"),
            "split_sha256": _sha(split_path),
            "selection_parameters": dict(hvg.HVG_PARAMETERS),
            "selection_source": hvg.AUTHOR_SOURCE,
            "selection_source_sha256": _sources(),
            "seed": int(config.get("seed", 42))}


def _plan(config, root, protocol):
    if protocol not in config.get("evaluation", {}).get("protocols", []):
        raise ValueError(f"Unknown configured HVG protocol {protocol!r}")
    path = _path(config, root, "splits_dir", "data/splits") / f"{protocol}.json"
    payload = _json(path)
    if payload.get("scheme") != protocol or payload.get("schema_version") != 1:
        raise ValueError("HVG split protocol/schema differs from its manifest")
    cells = payload.get("cell_ids", [])
    folds = payload.get("folds", [])
    if (not cells or len(set(cells)) != len(cells) or payload.get("n_splits") != len(folds)
            or len(folds) < 2):
        raise ValueError("Invalid HVG split universe or fold count")
    all_test = []
    for split in folds:
        train, test = split["train_ids"], split["test_ids"]
        if (not train or not test or len(set(train)) != len(train) or len(set(test)) != len(test)
                or set(train) & set(test) or set(train + test) != set(cells)):
            raise ValueError("HVG split must uniquely partition every canonical cell")
        all_test.extend(test)
    if len(set(all_test)) != len(all_test):
        raise ValueError("HVG split repeats test cells across folds")
    if set(cells) - set(all_test) != set(payload.get("never_tested_ids", [])):
        raise ValueError("HVG split test coverage differs from its manifest")
    if protocol != "author_historical" and set(all_test) != set(cells):
        raise ValueError("Corrected HVG splits must test every cell")
    with np.load(_prepared(config, root) / "expression.npz", allow_pickle=False) as archive:
        expression_ids = archive["cell_ids"].astype(str).tolist()
    if cells != expression_ids:
        raise ValueError("HVG split canonical cell order differs from expression.npz")
    return payload


def _split(config, root, protocol, fold):
    plan = _plan(config, root, protocol)
    if not isinstance(fold, int) or isinstance(fold, bool) or not 0 <= fold < len(plan["folds"]):
        raise ValueError(f"HVG fold {fold!r} is outside {protocol}")
    split = plan["folds"][fold]
    return plan, split["train_ids"], split["test_ids"]


def _panel_context(config, root, protocol, fold, train_ids, test_ids):
    return {"selector_context": _selector_context(config, root, protocol), "fold": fold,
            "train_ids": list(train_ids), "test_ids": list(test_ids),
            "input_order": "training_then_test", "counts_kind": "original raw exon+intron X",
            "labels_used": False}


def _raw_subset(config, root, ids, genes, source=None):
    import anndata as ad

    raw = source if source is not None else ad.read_h5ad(_prepared(config, root) / "counts.h5ad")
    if "species" in raw.obs and not raw.obs["species"].astype(str).eq("mouse").all():
        raise ValueError("Original HVG counts must contain mouse cells only")
    if not raw.obs_names.is_unique or not raw.var_names.is_unique:
        raise ValueError("Original counts contain duplicate cell/gene identifiers")
    with np.load(_prepared(config, root) / "expression.npz", allow_pickle=False) as archive:
        if (set(raw.obs_names.astype(str)) != set(archive["cell_ids"].astype(str)) or
                set(raw.var_names.astype(str)) != set(archive["gene_names"].astype(str))):
            raise ValueError("Original counts and expression cell/gene universes differ")
    selected = raw[ids, genes]
    counts = sparse.csr_matrix(selected.X, dtype=np.float64)
    counts.eliminate_zeros()
    if (not np.isfinite(counts.data).all() or (counts.data < 0).any() or
            not np.allclose(counts.data, np.round(counts.data), atol=1e-6, rtol=0)):
        raise ValueError("HVG panel X must retain raw finite nonnegative integer counts")
    if (np.asarray(counts.sum(axis=1)).ravel() <= 0).any():
        raise ValueError("A cell has no positive expression in its training-fold HVG panel")
    return ad.AnnData(X=counts, obs=pd.DataFrame({"species": "mouse"}, index=pd.Index(ids)),
                      var=pd.DataFrame(index=pd.Index(genes)))


def _read_panel(config, root, protocol, fold, *, verify_raw=False):
    directory = _directory(config, root, protocol, fold)
    counts_path, selection_path = directory / "counts.h5ad", directory / "selection.json"
    if not counts_path.is_file() or not selection_path.is_file():
        raise ValueError(f"Missing or partial immutable HVG panel: {directory}")
    plan, train_ids, test_ids = _split(config, root, protocol, fold)
    selection = _json(selection_path)
    expected = _panel_context(config, root, protocol, fold, train_ids, test_ids)
    if (selection.get("schema_version") != 1 or selection.get("kind") != "training_fold_native_hvg_panel" or
            selection.get("protocol") != protocol or selection.get("fold") != fold or
            selection.get("labels_used") is not False or selection.get("panel_context") != expected or
            selection.get("panel_context_sha256") != _digest(expected) or
            selection.get("counts_sha256") != _sha(counts_path)):
        raise ValueError(f"Stale or changed HVG panel context/checksum: {directory}")
    genes = selection.get("selector_audit", {}).get("genes", [])
    if (not genes or len(set(genes)) != len(genes) or selection.get("genes") != genes or
            selection.get("n_selected_genes") != len(genes)):
        raise ValueError("HVG panel genes differ from the native selector audit")
    data = embeddings._load_prepared(counts_path)
    if (data.obs_names.astype(str).tolist() != train_ids + test_ids or
            data.var_names.astype(str).tolist() != genes or list(data.obs.columns) != ["species"] or
            list(data.var.columns) or list(data.layers) or data.raw is not None or data.uns or data.obsm or data.varm):
        raise ValueError("HVG panel has wrong cell/gene order or contains non-species metadata")
    if verify_raw:
        _, _, native_audit, _ = hvg.build_features(config, _prepared(config, root), "hvg", train_ids, test_ids)
        if selection.get("selector_audit") != native_audit:
            raise ValueError("HVG panel does not exactly match the native training-only selector")
        expected_data = _raw_subset(config, root, train_ids + test_ids, genes)
        if (sparse.csr_matrix(data.X) != expected_data.X).nnz:
            raise ValueError("HVG panel differs from original raw exon+intron counts")
    return data, selection, plan, train_ids, test_ids


def prepare_hvg_panels(config: dict, root: Path) -> dict:
    """Prepare every training-only selected raw-count panel, reusing exact matches."""
    import anndata as ad

    root = Path(root).resolve()
    records = []
    source = None
    for protocol in config["evaluation"]["protocols"]:
        plan = _plan(config, root, protocol)
        for fold, split in enumerate(plan["folds"]):
            directory = _directory(config, root, protocol, fold)
            counts_path, selection_path = directory / "counts.h5ad", directory / "selection.json"
            if counts_path.exists() or selection_path.exists():
                _, selection, _, _, _ = _read_panel(config, root, protocol, fold, verify_raw=True)
                reused = True
            else:
                if directory.exists() and any(directory.iterdir()):
                    raise ValueError(f"Partial immutable HVG panel directory: {directory}")
                train_ids, test_ids = split["train_ids"], split["test_ids"]
                _, _, audit, _ = hvg.build_features(config, _prepared(config, root), "hvg", train_ids, test_ids)
                genes = audit["genes"]
                if source is None:
                    source = ad.read_h5ad(_prepared(config, root) / "counts.h5ad")
                data = _raw_subset(config, root, train_ids + test_ids, genes, source)
                context = _panel_context(config, root, protocol, fold, train_ids, test_ids)
                directory.mkdir(parents=True, exist_ok=True)
                temporary = directory / f"counts.{uuid.uuid4().hex}.tmp.h5ad"
                data.write_h5ad(temporary)
                os.replace(temporary, counts_path)
                selection = {"schema_version": 1, "kind": "training_fold_native_hvg_panel",
                             "protocol": protocol, "fold": fold, "genes": genes,
                             "n_selected_genes": len(genes), "selector_audit": audit,
                             "panel_context": context, "panel_context_sha256": _digest(context),
                             "counts_sha256": _sha(counts_path), "labels_used": False}
                atomic_json(selection_path, selection)
                reused = False
            records.append({"protocol": protocol, "fold": fold, "path": str(counts_path),
                            "genes": selection["n_selected_genes"], "reused": reused})
    return {"panels": records, "n_panels": len(records), "labels_used": False}


def check_hvg_panels(config: dict, root: Path) -> dict:
    """Validate every immutable panel and its exact original raw count values."""
    records = []
    for protocol in config["evaluation"]["protocols"]:
        for fold in range(len(_plan(config, root, protocol)["folds"])):
            data, selection, _, _, _ = _read_panel(config, root, protocol, fold, verify_raw=True)
            records.append({"protocol": protocol, "fold": fold, "cells": data.n_obs,
                            "genes": selection["n_selected_genes"], "valid": True})
    return {"panels": records, "n_panels": len(records), "valid": True}


def _stable_context(name, config, root, protocol, device, *, runtime_from=None):
    prepared = _prepared(config, root) / "counts.h5ad"
    with _encoder_hash_cache():
        encoder_context = embeddings._embedding_context(name, config["embeddings"][name], Path(root).resolve(),
                                                        prepared, str(device))
    if runtime_from is not None:
        # Evaluators intentionally use a different environment than UCE's Python
        # 3.11 encoder environment. Stored runtime is verified structurally, not
        # compared against the consumer interpreter or installed packages.
        previous = runtime_from["encoder_context"]
        if not isinstance(previous.get("python"), str) or not isinstance(previous.get("packages"), dict):
            raise ValueError("HVG embedding audit lacks its encoder runtime provenance")
        encoder_context["python"] = previous["python"]
        encoder_context["packages"] = previous["packages"]
    return {"schema_version": 1, "kind": "training_fold_native_hvg_embedding",
            "encoder": name, "selector_context": _selector_context(config, root, protocol),
            "encoder_context": encoder_context, "seed": int(config.get("seed", 42)),
            "dimensions": DIMENSIONS[name], "input_counts": "original raw exon+intron X"}


def _coverage(data, name, config, root):
    """Require positive in-vocabulary expression for every input cell."""
    settings = config["embeddings"][name]
    root = Path(root).resolve()
    if name == "genept":
        names, _ = embeddings.load_genept_vectors(resolve(root, settings["vectors_path"]))
        vectors = set(names)
        frame = pd.read_csv(resolve(root, settings["orthologs_path"]), sep="\t", dtype=str).fillna("")
        if not {"mouse_symbol", "human_symbol"} <= set(frame):
            raise ValueError("GenePT ortholog table lacks required columns")
        if frame.mouse_symbol.duplicated().any() or frame.human_symbol.duplicated().any():
            raise ValueError("GenePT ortholog table must be one-to-one")
        genes = set(frame.loc[frame.human_symbol.isin(vectors), "mouse_symbol"])
    elif name == "nicheformer":
        import anndata as ad
        checkpoint = resolve(root, settings["checkpoint_dir"])
        reference = ad.read_h5ad(checkpoint / "model.h5ad", backed="r")
        native = set(reference.var_names.astype(str))
        if settings.get("gene_mapping_path"):
            frame = pd.read_csv(resolve(root, settings["gene_mapping_path"]), sep="\t", dtype=str).fillna("")
            if "mouse_symbol" not in frame or frame.mouse_symbol.duplicated().any():
                raise ValueError("Nicheformer mapping must have unique mouse_symbol rows")
            genes = {row["mouse_symbol"] for row in frame.to_dict("records")
                     if any(row.get(key, "") in native for key in ("model_gene_id", "mouse_ensembl", "human_ensembl"))}
        else:
            genes = set()
            for column in ("mouse_symbol", "gene_symbols", "gene_symbol", "feature_name", "gene_name"):
                if column in reference.var:
                    aliases = reference.var[column].astype(str)
                    genes.update(aliases[~aliases.duplicated(keep=False)])
        if getattr(reference, "file", None) is not None:
            reference.file.close()
    else:
        # The vendor filters mouse symbols against the chromosome table and the
        # mouse protein dictionary before tokenization; validate that same set.
        import torch
        resources = resolve(root, settings["resources_dir"])
        chrom = pd.read_csv(resources / "species_chrom.csv")
        if "species" in chrom:
            chrom = chrom.loc[chrom["species"].astype(str).eq("mouse")]
        if "gene_symbol" not in chrom or "species" not in chrom:
            raise ValueError("UCE chromosome vocabulary requires species/gene_symbol")
        chrom_genes = set(chrom["gene_symbol"].astype(str).str.upper())
        # These are the exact native filename and upper-case lookup used by
        # UCE data_proc/data_utils.py:get_species_to_pe/generate_idxs.
        protein_path = resources / "protein_embeddings" / "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt"
        protein = torch.load(protein_path, map_location="cpu", weights_only=False)
        if not isinstance(protein, dict):
            raise ValueError("UCE mouse protein vocabulary must be a gene dictionary")
        covered = chrom_genes & {str(gene).upper() for gene in protein}
        genes = {str(gene) for gene in data.var_names if str(gene).upper() in covered}
    columns = [i for i, gene in enumerate(data.var_names) if str(gene) in genes]
    expressed = np.asarray(data.X[:, columns].sum(axis=1)).ravel() if columns else np.zeros(data.n_obs)
    if (expressed <= 0).any():
        raise ValueError(f"{name}: {(expressed <= 0).sum()} cells lack positive in-vocabulary HVG expression")
    return {"matched_genes": len(columns), "cells_with_positive_in_vocab_expression": data.n_obs,
            "min_in_vocab_counts": float(expressed.min()), "all_cells_covered": True}


def _embedding_paths(config, root, name, protocol, fold):
    directory = _directory(config, root, protocol, fold)
    return directory / f"{name}.npz", directory / f"{name}.audit.json"


def _validate_embedding(config, root, name, protocol, fold, *, runtime=False, device=None):
    data, selection, plan, train_ids, test_ids = _read_panel(config, root, protocol, fold)
    output, audit_path = _embedding_paths(config, root, name, protocol, fold)
    if not output.is_file() or not audit_path.is_file():
        raise ValueError(f"Missing or partial immutable HVG {name} embedding: {output.parent}")
    audit = _json(audit_path)
    stable = audit.get("stable_context")
    if not isinstance(stable, dict) or not isinstance(stable.get("encoder_context"), dict):
        raise ValueError("HVG embedding has no stable fold-family context")
    expected = _stable_context(name, config, root, protocol, device if runtime else audit.get("device"),
                               runtime_from=None if runtime else stable)
    if (stable != expected or audit.get("stable_context_sha256") != _digest(stable) or
            audit.get("panel_context_sha256") != selection["panel_context_sha256"] or
            audit.get("selection_sha256") != _sha(output.parent / "selection.json") or
            audit.get("counts_sha256") != selection["counts_sha256"] or
            audit.get("output_sha256") != _sha(output) or audit.get("embedding") != name or
            audit.get("protocol") != protocol or audit.get("fold") != fold or
            audit.get("train_ids") != train_ids or audit.get("test_ids") != test_ids or
            audit.get("rows") != len(plan["cell_ids"]) or audit.get("dimensions") != DIMENSIONS[name] or
            audit.get("labels_used") is not False or audit.get("frozen") is not True or
            audit.get("coverage", {}).get("all_cells_covered") is not True or
            audit.get("coverage", {}).get("cells_with_positive_in_vocab_expression") != data.n_obs):
        raise ValueError(f"Stale or changed HVG {name} context/checksum/scope: {output}")
    with np.load(output, allow_pickle=False) as archive:
        values, ids = archive["values"], archive["cell_ids"].astype(str)
        if (values.shape != (data.n_obs, DIMENSIONS[name]) or not np.isfinite(values).all() or
                ids.tolist() != plan["cell_ids"] or len(set(ids)) != len(ids)):
            raise ValueError("HVG embedding has invalid finite dimensions or canonical cell IDs")
    return output, audit_path, values.astype(np.float32), ids, audit


def generate_hvg_embedding(name: str, config: dict, *, root: Path, protocol: str,
                           fold: int, device: str, offline: bool = True) -> EmbeddingResult:
    """Encode one panel in training-first order and save canonical cell-ID order."""
    if name not in ENCODERS:
        raise ValueError(f"Unknown HVG encoder {name!r}")
    if not offline:
        raise ValueError("Native-HVG encoding requires frozen offline resources")
    root = Path(root).resolve()
    started = time.perf_counter()
    data, selection, plan, train_ids, test_ids = _read_panel(config, root, protocol, fold, verify_raw=True)
    output, audit_path = _embedding_paths(config, root, name, protocol, fold)
    if output.exists() or audit_path.exists():
        _, _, _, _, audit = _validate_embedding(config, root, name, protocol, fold, runtime=True, device=device)
        return EmbeddingResult(output, audit)
    embeddings._offline_environment(root / "cache", offline)
    stable = _stable_context(name, config, root, protocol, device)
    coverage = _coverage(data, name, config, root)
    settings = config["embeddings"][name]
    with _encoder_hash_cache():
        if name == "genept":
            values, audit = embeddings._genept(data, settings, root)
        elif name == "nicheformer":
            values, audit = embeddings._nicheformer(data, settings, root, device, offline)
        else:
            work = output.parent / f"uce_work_{uuid.uuid4().hex}"
            values, audit = embeddings._uce(data, settings, root, device, offline, work)
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (data.n_obs, DIMENSIONS[name]) or not np.isfinite(values).all():
        raise ValueError(f"Expected finite native {name} output shape {(data.n_obs, DIMENSIONS[name])}")
    index = {cell: i for i, cell in enumerate(data.obs_names.astype(str))}
    canonical = plan["cell_ids"]
    values = values[[index[cell] for cell in canonical]]
    audit.update(embedding=name, representation=f"hvg_{name}", protocol=protocol, fold=fold,
                 train_ids=train_ids, test_ids=test_ids, device=str(device), offline=True,
                 stable_context=stable, stable_context_sha256=_digest(stable),
                 panel_context_sha256=selection["panel_context_sha256"],
                 selection_sha256=_sha(output.parent / "selection.json"),
                 counts_sha256=selection["counts_sha256"], coverage=coverage,
                 elapsed_seconds=time.perf_counter() - started, input_order="training_then_test",
                 output_order="canonical_expression_cell_ids")
    return embeddings._save(name, values, canonical, output.parent, audit)


def _artifacts(output, audit_path):
    return {key: {"path": str(path), "sha256": _sha(path)} for key, path in
            (("selection", output.parent / "selection.json"), ("counts", output.parent / "counts.h5ad"),
             ("embedding", output), ("audit", audit_path))}


def load_hvg_features(config: dict, root: Path, representation: str, protocol: str,
                      fold: int, train_ids: list[str], test_ids: list[str]):
    """Load fold-local embeddings with explicit ID alignment and immutable provenance."""
    if representation not in HVG_EMBEDDING_REPRESENTATIONS:
        raise ValueError(f"Unknown HVG embedding representation {representation!r}")
    name = representation.removeprefix("hvg_")
    _, expected_train, expected_test = _split(config, root, protocol, fold)
    if list(train_ids) != expected_train or list(test_ids) != expected_test:
        raise ValueError("Requested HVG train/test IDs differ from the exact outer split")
    output, audit_path, values, ids, encoding_audit = _validate_embedding(config, root, name, protocol, fold)
    artifacts = _artifacts(output, audit_path)
    scope = {"kind": "fold_embedding", "encoder": name, "protocol": protocol, "fold": fold,
             "stable_context": copy.deepcopy(encoding_audit["stable_context"]),
             "stable_context_sha256": encoding_audit["stable_context_sha256"],
             "fold_artifacts": artifacts, "train_ids": list(train_ids), "test_ids": list(test_ids)}
    audit = {"kind": "training_fold_native_hvg_embedding", "labels_used": False,
             "frozen": True, "n_features": values.shape[1], "feature_scope": scope,
             "genes": _json(output.parent / "selection.json")["genes"],
             "coverage": encoding_audit["coverage"]}
    index = {cell: position for position, cell in enumerate(ids)}
    inputs = {entry["path"]: entry["sha256"] for entry in artifacts.values()}
    return (values[[index[cell] for cell in train_ids]], values[[index[cell] for cell in test_ids]],
            audit, inputs)


def validate_hvg_feature_scope(config: dict, root: Path, feature_audit: dict, *, protocol: str,
                               fold: int, train_ids: list[str], test_ids: list[str]) -> str:
    """Validate saved evaluation feature scope without requiring encoder package versions."""
    scope = feature_audit.get("feature_scope", {})
    name = scope.get("encoder")
    if (scope.get("kind") != "fold_embedding" or name not in ENCODERS or
            scope.get("protocol") != protocol or scope.get("fold") != fold or
            scope.get("train_ids") != list(train_ids) or scope.get("test_ids") != list(test_ids)):
        raise ValueError("HVG evaluation feature scope differs from the requested fold")
    _, _, current, _ = load_hvg_features(config, root, f"hvg_{name}", protocol, fold, train_ids, test_ids)
    if scope != current["feature_scope"]:
        raise ValueError("HVG evaluation feature artifact hashes or stable context changed")
    return scope["stable_context_sha256"]


def check_hvg_embeddings(config: dict, root: Path, *, encoder=None, first_only=False) -> dict:
    names = (encoder,) if encoder else ENCODERS
    if any(name not in ENCODERS for name in names):
        raise ValueError(f"Unknown HVG encoder {encoder!r}")
    protocols = list(config["evaluation"]["protocols"])
    if first_only:
        protocols = ["cell_kfold"] if "cell_kfold" in protocols else protocols[:1]
    records = []
    for protocol in protocols:
        folds = [0] if first_only else range(len(_plan(config, root, protocol)["folds"]))
        for fold in folds:
            for name in names:
                output, _, _, _, audit = _validate_embedding(config, root, name, protocol, fold)
                records.append({"encoder": name, "protocol": protocol, "fold": fold,
                                "path": str(output), "dimensions": audit["dimensions"],
                                "elapsed_seconds": audit["elapsed_seconds"], "device": audit["device"], "valid": True})
    return {"embeddings": records, "n_embeddings": len(records), "valid": True}


def encoding_budget(config: dict, root: Path) -> dict:
    """Gate the remaining two-encoder GPU sweep using measured first-cell-fold jobs."""
    protocols = config["evaluation"]["protocols"]
    if "cell_kfold" not in protocols:
        raise ValueError("Encoding budget requires the first cell_kfold pilot")
    pilot = check_hvg_embeddings(config, root, first_only=True)
    measured = {row["encoder"]: row for row in pilot["embeddings"]}
    if any(not str(measured[name]["device"]).startswith("cuda") for name in ("uce", "nicheformer")):
        raise ValueError("GPU encoding budget requires CUDA-measured UCE and Nicheformer pilots")
    remaining_counts = dict.fromkeys(ENCODERS, 0)
    remaining_jobs = []
    for protocol in protocols:
        for fold in range(len(_plan(config, root, protocol)["folds"])):
            for name in ENCODERS:
                if protocol == "cell_kfold" and fold == 0:
                    continue
                output, audit = _embedding_paths(config, root, name, protocol, fold)
                if output.exists() or audit.exists():
                    _validate_embedding(config, root, name, protocol, fold)
                else:
                    remaining_counts[name] += 1
                    remaining_jobs.append({"encoder": name, "protocol": protocol, "fold": fold})
    seconds = {name: float(measured[name]["elapsed_seconds"]) for name in ENCODERS}
    if any(not np.isfinite(value) or value <= 0 for value in seconds.values()):
        raise ValueError("Encoding pilot elapsed times must be finite and positive")
    gpu_hours = sum(2 * remaining_counts[name] * seconds[name] / 3600 for name in ("uce", "nicheformer"))
    cpu_hours = 2 * remaining_counts["genept"] * seconds["genept"] / 3600
    budget = config.get("cluster", {}).get("remaining_gpu_hours")
    if budget is not None and (not np.isfinite(float(budget)) or float(budget) < 0):
        raise ValueError("remaining_gpu_hours must be a finite nonnegative number")
    enough = budget is not None and gpu_hours <= float(budget)
    return {"schema_version": 1, "pilot_protocol": "cell_kfold", "pilot_fold": 0,
            "safety_multiplier": 2, "pilot_elapsed_seconds": seconds,
            "remaining_jobs_per_encoder": remaining_counts, "remaining_jobs": remaining_jobs,
            "remaining_gpu_hours_estimate": gpu_hours, "remaining_cpu_hours_estimate": cpu_hours,
            "remaining_cpu_wall_hours_estimate": cpu_hours,
            "cpu_cores_per_job": int(config.get("cluster", {}).get("encoder_cpus", 16)),
            "remaining_cpu_core_hours_estimate": cpu_hours * int(config.get("cluster", {}).get("encoder_cpus", 16)),
            "remaining_gpu_hours": budget, "within_budget": enough, "approved": enough,
            "budget_status": "unset" if budget is None else "sufficient" if enough else "insufficient"}
