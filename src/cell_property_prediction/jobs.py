"""Scheduler-independent job manifests and conservative allocation estimates."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from .config import atomic_json, compute_config, file_sha256, resolve

FIELDS = ("protocol", "representation", "predictor", "fold")


def create_jobs(config: dict, root: Path, stage: str, output: Path) -> list[dict]:
    if stage == "simple":
        representations, predictors = ["pca128", "ion_channels", "authors25"], ["tabpfn"]
    elif stage == "hvg":
        # An additive stage keeps the original configuration/result provenance
        # intact while evaluating both new inputs with every foundation model.
        representations, predictors = ["hvg", "hvg_pca128"], ["tabpfn", "limix", "kumo"]
    elif stage in {"full", "pending"}:
        # Include the two HVG inputs without changing the frozen base config.
        representations = list(dict.fromkeys([*config["evaluation"]["representations"],
                                               "hvg", "hvg_pca128"]))
        predictors = [p for p in config["evaluation"]["predictors"] if p in {"tabpfn", "limix", "kumo"}]
    elif stage == "baselines":
        representations = config["evaluation"]["representations"]
        predictors = ["ridge", "mean"]
    elif stage == "rrr":
        representations, predictors = ["raw1000"], ["sparse_rrr"]
    else:
        raise ValueError(f"Unknown job stage: {stage}")
    jobs = []
    for protocol in config["evaluation"]["protocols"]:
        split_file = resolve(root, config["paths"]["splits_dir"]) / f"{protocol}.json"
        if not split_file.is_file():
            raise FileNotFoundError(f"Prepare splits first: {split_file}")
        split = json.loads(split_file.read_text(encoding="utf-8"))
        for representation in representations:
            for predictor in predictors:
                if stage == "pending" and predictor == "tabpfn" and representation in {
                    "pca128", "ion_channels", "authors25"
                }:
                    continue
                for fold in range(len(split["folds"])):
                    jobs.append(dict(zip(FIELDS, (protocol, representation, predictor, fold))))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(jobs)
    atomic_json(output.with_suffix(".json"), {"stage": stage, "jobs": len(jobs),
                "targets_per_job": 17, "target_fits": 17 * len(jobs), "fields": FIELDS})
    return jobs


def read_job(path: Path, index: int) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if index < 0 or index >= len(rows):
        raise IndexError(f"Array index {index} is outside 0..{len(rows)-1}")
    row = rows[index]
    if set(row) != set(FIELDS):
        raise ValueError("Unexpected job manifest columns")
    row["fold"] = int(row["fold"])
    return row


def compute_fingerprint(config: dict, root: Path) -> str:
    """Use the same immutable compute context for pilot and budget validation."""
    context = compute_config(config)
    # The actual allocation is supplied after the measurements; it is not a
    # model or preprocessing choice and must not invalidate the pilot.
    paths = list((root / "src" / "cell_property_prediction").glob("*.py"))
    prepared = resolve(root, config["paths"]["prepared_dir"])
    paths.extend(prepared / name for name in ("manifest.json", "expression.npz", "counts.h5ad",
                 "metadata.csv", "targets_raw.csv", "targets_author.csv"))
    paths.append(resolve(root, config["paths"]["cache_dir"]) / "model_manifest.json")
    paths.extend(resolve(root, config["paths"]["splits_dir"]).glob("*.json"))
    embeddings = resolve(root, config["paths"]["embeddings_dir"])
    paths.extend(embeddings.glob("*.npz"))
    paths.extend(embeddings.glob("*.audit.json"))
    files = {str(path.resolve()): file_sha256(path) for path in paths if path.is_file()}
    return hashlib.sha256(json.dumps({"config": context, "files": files}, sort_keys=True).encode()).hexdigest()


def estimate_jobs(config: dict, jobs: list[dict], pilot_path: Path, *, root: Path) -> dict:
    pilots = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilots.get("fingerprint") != compute_fingerprint(config, root):
        raise ValueError("Pilot provenance differs from current inputs/code/configuration; measure a new pilot")
    times = {(r["representation"], r["predictor"]): r for r in pilots["runs"]}
    hours = 0.0
    missing = []
    for job in jobs:
        if job["predictor"] not in {"tabpfn", "limix", "kumo"}:
            continue
        record = times.get((job["representation"], job["predictor"]))
        if record is None:
            missing.append([job["representation"], job["predictor"]])
            continue
        if not str(record.get("device", "")).startswith("cuda"):
            raise ValueError("GPU budget requires a CUDA pilot for every requested model/representation")
        # Include startup/feature preparation, scale the pilot target count and a 2x margin.
        hours += 2 * record["elapsed_seconds"] * 17 / record["targets"] / 3600
    if missing:
        raise ValueError(f"Missing pilot measurements: {sorted(set(map(tuple, missing)))}")
    remaining = config["cluster"].get("remaining_gpu_hours")
    return {"estimated_gpu_hours_with_2x_margin": hours, "jobs": len(jobs),
            "remaining_gpu_hours": remaining,
            "within_budget": remaining is not None and math.isfinite(float(remaining)) and hours <= float(remaining)}
