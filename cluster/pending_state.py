"""Protect completed simple benchmarks while extending the experiment matrix.

Run with the core environment. This helper never downloads assets or runs models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from cell_property_prediction.config import atomic_json, compute_config, file_sha256, load_config, resolve

SIMPLE = ("pca128", "ion_channels", "authors25")
PROTOCOLS = ("cell_kfold", "mouse_group_kfold")
EMBEDDINGS = {"genept": 3072, "uce": 1280, "nicheformer": 512}


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(compute_config(config), sort_keys=True, default=str).encode()).hexdigest()


def _targets(config: dict, root: Path) -> tuple[list[str], set[str]]:
    source = resolve(root, config["paths"]["prepared_dir"]) / "targets_raw.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    targets = header[1:]
    cells = [row[0] for row in rows]
    if not targets or len(set(targets)) != len(targets) or not cells or len(set(cells)) != len(cells):
        raise ValueError("Prepared targets require unique cell IDs and target columns")
    if len(cells) != int(config["data"]["expected_cohort_cells"]):
        raise ValueError("Prepared cohort differs from the configured cohort size")
    if config["evaluation"].get("targets") not in (None, targets):
        raise ValueError("Pending production requires the complete prepared target set")
    return targets, set(cells)


def _collect_baselines(config: dict, root: Path) -> tuple[dict[str, str], int]:
    """Verify completed folds and protect their recorded inputs and predictions."""
    if set(config["evaluation"]["protocols"]) != set(PROTOCOLS):
        raise ValueError("Pending continuation expects both cell and mouse CV protocols")
    targets, cells = _targets(config, root)
    protected: dict[str, str] = {}
    config_hash = _config_hash(config)
    fold_count = 0

    def protect(path: Path, expected: str | None = None) -> None:
        path = path.resolve()
        key = str(path)
        if not path.is_file():
            raise ValueError(f"Required baseline file is missing: {path}")
        if key not in protected:
            protected[key] = file_sha256(path)
        if expected is not None and protected[key] != expected:
            raise ValueError(f"Completed baseline input or prediction changed: {path}")

    prepared = resolve(root, config["paths"]["prepared_dir"])
    for name in ("manifest.json", "expression.npz", "counts.h5ad", "metadata.csv",
                 "targets_raw.csv", "targets_author.csv"):
        protect(prepared / name)
    # Preserve any original downloads still present, while allowing a prepared-only checkout.
    from cell_property_prediction.assets import SOURCE_FILES
    raw_dir = resolve(root, config["paths"]["raw_dir"])
    for name in (*SOURCE_FILES, "download_manifest.json"):
        raw_path = raw_dir / name
        if raw_path.is_file():
            protect(raw_path)
    splits_dir = resolve(root, config["paths"]["splits_dir"])
    for path in sorted(splits_dir.glob("*.json")):
        protect(path)
    for protocol in PROTOCOLS:
        split_path = splits_dir / f"{protocol}.json"
        plan = _json(split_path)
        folds = plan["folds"]
        if len(folds) != int(config["evaluation"]["n_splits"]):
            raise ValueError(f"Unexpected number of folds: {split_path}")
        seen: set[str] = set()
        for fold in folds:
            train, test = set(fold["train_ids"]), set(fold["test_ids"])
            if not train or not test or train & test or train | test != cells or seen & test:
                raise ValueError(f"Invalid or incomplete outer-fold coverage: {split_path}")
            if len(train) != len(fold["train_ids"]) or len(test) != len(fold["test_ids"]):
                raise ValueError(f"Duplicate IDs in outer split: {split_path}")
            seen.update(test)
        if seen != cells:
            raise ValueError(f"Some cells are never tested: {split_path}")
        for representation in SIMPLE:
            for index, fold in enumerate(folds):
                directory = (resolve(root, config["paths"]["results_dir"]) /
                             protocol / representation / "tabpfn" / f"fold_{index:02d}")
                path = directory / "manifest.json"
                if not path.is_file():
                    raise ValueError(f"Finish the three simple benchmarks first; missing {path}")
                manifest = _json(path)
                identity = (manifest.get("protocol"), manifest.get("representation"),
                            manifest.get("predictor"), manifest.get("fold"))
                if identity != (protocol, representation, "tabpfn", index):
                    raise ValueError(f"Baseline identity mismatch: {path}")
                if (manifest.get("status") != "complete" or manifest.get("pilot_target_subset")
                        or manifest.get("target_names") != targets):
                    raise ValueError(f"Baseline fold is incomplete or only a pilot: {path}")
                outcomes = manifest.get("targets", [])
                if (len(outcomes) != len(targets)
                        or {r.get("target") for r in outcomes} != set(targets)
                        or any(r.get("status") != "complete" for r in outcomes)):
                    raise ValueError(f"Baseline fold lacks completed target fits: {path}")
                if manifest.get("config_sha256") != config_hash:
                    raise ValueError(f"Configuration differs from completed baselines: {path}")
                protect(path)
                protect(directory / "predictions.csv", manifest.get("prediction_sha256") or "missing")
                for artifact in sorted(directory.rglob("*")):
                    if artifact.is_file() and artifact.suffix in {".json", ".csv"}:
                        protect(artifact)
                for kind in ("input_sha256", "model_input_sha256"):
                    records = manifest.get(kind, {})
                    if not records:
                        raise ValueError(f"Baseline provenance lacks {kind}: {path}")
                    for name, expected in records.items():
                        protect(resolve(root, name), expected)
                expected_test = set(fold["test_ids"])
                observed: set[tuple[str, str]] = set()
                with (directory / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                    for row in csv.DictReader(stream):
                        key = (row["cell_id"], row["target"])
                        if (key in observed or key[0] not in expected_test or key[1] not in targets
                                or not math.isfinite(float(row["y_pred"]))):
                            raise ValueError(f"Invalid baseline predictions: {directory}")
                        if (row["protocol"], row["representation"], row["predictor"], int(row["fold"])) != identity:
                            raise ValueError(f"Baseline prediction identity mismatch: {directory}")
                        observed.add(key)
                if len(observed) != len(expected_test) * len(targets):
                    raise ValueError(f"Baseline prediction coverage is incomplete: {directory}")
                fold_count += 1
    return protected, fold_count


def check_embeddings(config: dict, root: Path, names=None) -> dict[str, int]:
    import numpy as np

    _, cells = _targets(config, root)
    result = {}
    prepared_sha = file_sha256(resolve(root, config["paths"]["prepared_dir"]) / "counts.h5ad")
    for name in (names or EMBEDDINGS):
        dimension = EMBEDDINGS[name]
        path = resolve(root, config["paths"]["embeddings_dir"]) / f"{name}.npz"
        audit_path = path.with_suffix(".audit.json")
        if not path.is_file() or not audit_path.is_file():
            raise ValueError(f"Complete the {name} embedding job first: {path}")
        audit = _json(audit_path)
        if (audit.get("output_sha256") != file_sha256(path)
                or audit.get("labels_used") is not False or audit.get("frozen") is not True):
            raise ValueError(f"Embedding audit is invalid: {audit_path}")
        context = audit.get("context", {})
        context_sha = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (audit.get("prepared_sha256") != prepared_sha
                or context.get("prepared_sha256") != prepared_sha
                or context.get("embedding") != name
                or context.get("config") != config["embeddings"][name]
                or audit.get("context_sha256") != context_sha):
            raise ValueError(f"Embedding input or configuration provenance differs: {audit_path}")
        with np.load(path, allow_pickle=False) as archive:
            values, ids = archive["values"], archive["cell_ids"].astype(str)
            if (values.shape != (len(cells), dimension) or len(ids) != len(cells)
                    or len(set(ids)) != len(ids) or set(ids) != cells
                    or not np.isfinite(values).all()):
                raise ValueError(f"Embedding dimensions, IDs or values are invalid: {path}")
        result[name] = dimension
    return result


def check(config: dict, root: Path, state_path: Path, *, embeddings: bool = False, names=None) -> dict:
    if not state_path.is_file():
        raise ValueError("Baseline snapshot is missing; run zsh cluster/pending.sh setup first")
    state = _json(state_path)
    if state.get("format_version") != 1 or state.get("root") != str(root):
        raise ValueError("Snapshot belongs to another project directory or unsupported format")
    if state.get("config_sha256") != _config_hash(config):
        raise ValueError("Compute configuration changed; preserve the completed baseline configuration")
    for section in ("protected_sha256", "archive_sha256"):
        for name, expected in state[section].items():
            path = Path(name)
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"Protected baseline file changed or is missing: {path}")
    cache_manifest = resolve(root, config["paths"]["cache_dir"]) / "model_manifest.json"
    if _json(cache_manifest).get("tabpfn") != state["tabpfn_resource"]:
        raise ValueError("The original TabPFN resource record changed")
    result = {"status": "verified", "baseline_folds": state["baseline_folds"],
              "state": str(state_path)}
    if embeddings or names:
        result["embeddings"] = check_embeddings(config, root, None if embeddings else names)
    return result


def snapshot(config: dict, root: Path, config_path: Path, state_path: Path) -> dict:
    if state_path.exists():
        return check(config, root, state_path)
    protected, folds = _collect_baselines(config, root)
    cache_manifest = resolve(root, config["paths"]["cache_dir"]) / "model_manifest.json"
    resources = _json(cache_manifest)
    if not isinstance(resources.get("tabpfn"), dict):
        raise ValueError("The existing TabPFN download manifest is required before continuation")
    archive = state_path.with_suffix("")
    copies = {"config.yaml": config_path, "model_manifest.json": cache_manifest}
    results = resolve(root, config["paths"]["results_dir"])
    old_pilot = results / "pilot" / "timings.json"
    if old_pilot.is_file():
        copies["pilot_timings.json"] = old_pilot
    reports = resolve(root, config["paths"].get("reports_dir", str(results / "reports")))
    for path in sorted(reports.glob("*")):
        if path.is_file() and path.suffix in {".csv", ".json"}:
            copies["reports/" + path.name] = path
    archived = {}
    for relative, source in copies.items():
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = file_sha256(source)
        if destination.exists() and file_sha256(destination) != expected:
            raise ValueError(f"An earlier snapshot file differs: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        archived[str(destination.resolve())] = expected
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unavailable"
    atomic_json(state_path, {
        "format_version": 1, "root": str(root), "config_sha256": _config_hash(config),
        "baseline_folds": folds, "protected_sha256": protected,
        "archive_sha256": archived, "tabpfn_resource": resources["tabpfn"],
        "git_head_at_snapshot": revision,
        "policy": "Keep completed fold artifacts and inputs immutable; additional models/embeddings may be added",
    })
    return check(config, root, state_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["snapshot", "check"])
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("CPP_ROOT", Path.cwd())))
    parser.add_argument("--config", default=os.environ.get("CPP_CONFIG", "configs/benchmark.yaml"))
    parser.add_argument("--state", default="results/pending_workflow/baseline-snapshot.json")
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--embedding", action="append", choices=list(EMBEDDINGS), default=[])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        config_path = resolve(root, args.config)
        config = load_config(config_path)
        state_path = resolve(root, args.state)
        if args.action == "snapshot":
            result = snapshot(config, root, config_path, state_path)
            if args.embeddings or args.embedding:
                result["embeddings"] = check_embeddings(config, root, None if args.embeddings else args.embedding)
        else:
            result = check(config, root, state_path, embeddings=args.embeddings, names=args.embedding)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        print(f"Pending workflow: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
