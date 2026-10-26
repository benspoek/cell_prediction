"""Command-line pipeline. Heavy libraries are imported only by the selected stage."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .config import atomic_json, load_config, resolve


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cellbench")
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--config", default="configs/benchmark.yaml")
    sub = p.add_subparsers(dest="command", required=True)
    prefetch = sub.add_parser("prefetch", help="Run on a cluster node with network access")
    prefetch.add_argument("--data", action="store_true")
    prefetch.add_argument("--models", nargs="*", choices=["tabpfn", "limix", "kumo", "genept", "uce", "nicheformer"], default=[])
    sub.add_parser("prepare")
    split = sub.add_parser("splits")
    split.add_argument("--historical", action="store_true")
    embed = sub.add_parser("embed")
    embed.add_argument("--encoder", required=True, choices=["genept", "uce", "nicheformer"])
    embed.add_argument("--device", default="cuda")
    embed.add_argument("--gene-set", choices=["full", "hvg"], default="full")
    embed.add_argument("--protocol", choices=["cell_kfold", "mouse_group_kfold"])
    embed.add_argument("--fold", type=int)
    panels = sub.add_parser("hvg-panels")
    panels.add_argument("--check", action="store_true")
    check = sub.add_parser("hvg-embedding-check")
    check.add_argument("--encoder", choices=["genept", "uce", "nicheformer"])
    check.add_argument("--first-only", action="store_true")
    encoder_budget = sub.add_parser("hvg-encoder-budget")
    encoder_budget.add_argument("--require-within-budget", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--protocol", choices=["cell_kfold", "mouse_group_kfold"], required=True)
    evaluate.add_argument("--representation", required=True)
    evaluate.add_argument("--predictor", required=True)
    evaluate.add_argument("--fold", type=int, required=True)
    evaluate.add_argument("--device", default="cuda")
    array = sub.add_parser("array-task")
    array.add_argument("--jobs", type=Path, required=True)
    array.add_argument("--index", type=int, default=None)
    array.add_argument("--device", default="cuda")
    jobs = sub.add_parser("jobs")
    jobs.add_argument("--stage", choices=["simple", "full", "pending", "hvg", "hvg_embeddings", "baselines", "rrr"], default="simple")
    jobs.add_argument("--output", type=Path, required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--predictors", nargs=1, choices=["tabpfn", "limix", "kumo"], required=True,
                       help="One model per invocation, using its isolated environment")
    pilot.add_argument("--representations", nargs="+", default=["pca128", "ion_channels", "authors25", "genept", "uce", "nicheformer", "hvg", "hvg_pca128", "hvg_genept", "hvg_uce", "hvg_nicheformer"])
    pilot.add_argument("--targets", type=int, default=2)
    pilot.add_argument("--output", type=Path,
                       help="Timing JSON path; predictions use its parent (default: results_dir/pilot/timings.json)")
    pilot.add_argument("--device", default="cuda", choices=["cuda", "cuda:0"],
                       help="GPU budget pilots require the allocated CUDA device")
    budget = sub.add_parser("budget")
    budget.add_argument("--jobs", type=Path, required=True)
    budget.add_argument("--pilot", type=Path, default=Path("results/pilot/timings.json"))
    budget.add_argument("--require-within-budget", action="store_true")
    history = sub.add_parser("historical")
    history.add_argument("--backend", choices=["sklearn_compatible", "original"], default="sklearn_compatible")
    sub.add_parser("report")
    sub.add_parser("preflight")
    return p


def evaluate_job(config, root, *, protocol, representation, predictor, fold, device):
    if predictor == "sparse_rrr":
        from .rrr import run_rrr_fold
        return run_rrr_fold(config, root=root, protocol=protocol, fold=fold)
    from .evaluation import evaluate_fold
    return evaluate_fold(config, root=root, protocol=protocol, representation=representation,
                         predictor=predictor, fold=fold, device=device)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.root.resolve()
    config = load_config(resolve(root, args.config))
    command = args.command
    if command == "prefetch":
        from .assets import fetch_data, prefetch_models
        result = {}
        if args.data:
            result["data"] = fetch_data(config, root)
        if args.models:
            result["models"] = prefetch_models(config, root, args.models)
        if not result:
            raise ValueError("Choose --data and/or --models")
    elif command == "prepare":
        from .data import load_source_tables, persist_prepared, prepare_cohort
        tables = load_source_tables(resolve(root, config["paths"]["raw_dir"]))
        expected = config["data"]
        if tables.exon_counts.shape != (expected["expected_raw_cells"], expected["expected_genes"]):
            raise ValueError(f"Unexpected raw shape: {tables.exon_counts.shape}")
        cohort = prepare_cohort(tables)
        if len(cohort.features) != expected["expected_cohort_cells"]:
            raise ValueError(f"Unexpected cohort: {cohort.cohort_audit}")
        result = persist_prepared(cohort, resolve(root, config["paths"]["prepared_dir"]))
    elif command == "splits":
        from .data import load_prepared
        from .splits import make_splits, save_splits
        cohort = load_prepared(resolve(root, config["paths"]["prepared_dir"]), load_counts=False)
        schemes = list(config["evaluation"]["protocols"])
        if args.historical:
            schemes.append("author_historical")
        result = {}
        for scheme in schemes:
            plan = make_splits(cohort.features.index, cohort.metadata, scheme,
                               config["evaluation"]["n_splits"], config["seed"])
            path = resolve(root, config["paths"]["splits_dir"]) / f"{scheme}.json"
            save_splits(plan, path)
            result[scheme] = {"path": path, "never_tested": len(plan.never_tested_ids)}
    elif command == "embed":
        if args.gene_set == "hvg":
            if args.protocol is None or args.fold is None:
                raise ValueError("--gene-set hvg requires --protocol and --fold")
            from .hvg_embeddings import generate_hvg_embedding
            result = generate_hvg_embedding(args.encoder, config, root=root,
                protocol=args.protocol, fold=args.fold, device=args.device, offline=True)
        else:
            if args.protocol is not None or args.fold is not None:
                raise ValueError("--protocol and --fold require --gene-set hvg")
            from .embeddings import generate_embedding
            result = generate_embedding(args.encoder, config["embeddings"][args.encoder],
                         resolve(root, config["paths"]["prepared_dir"]) / "counts.h5ad",
                         resolve(root, config["paths"]["embeddings_dir"]), device=args.device, root=root, offline=True)
    elif command == "hvg-panels":
        from .hvg_embeddings import prepare_hvg_panels, check_hvg_panels
        result = (check_hvg_panels if args.check else prepare_hvg_panels)(config, root)
    elif command == "hvg-embedding-check":
        from .hvg_embeddings import check_hvg_embeddings
        result = check_hvg_embeddings(config, root, encoder=args.encoder, first_only=args.first_only)
    elif command == "hvg-encoder-budget":
        from .hvg_embeddings import encoding_budget
        result = encoding_budget(config, root)
        if args.require_within_budget and not result["within_budget"]:
            print(json.dumps(result, indent=2, default=str))
            raise RuntimeError("Set the actual remaining GPU hours and complete first-fold encoder pilots before submission")
    elif command in {"evaluate", "array-task"}:
        if command == "array-task":
            from .jobs import read_job
            index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
            job = read_job(resolve(root, args.jobs), index)
        else:
            job = {key: getattr(args, key) for key in ("protocol", "representation", "predictor", "fold")}
        result = evaluate_job(config, root, **job, device=args.device)
    elif command == "jobs":
        from .jobs import create_jobs
        jobs = create_jobs(config, root, args.stage, resolve(root, args.output))
        result = {"jobs": len(jobs), "target_fits": 17 * len(jobs), "output": args.output}
    elif command == "pilot":
        from .data import TARGET_COLUMNS
        from .jobs import compute_fingerprint
        pilot_config = copy.deepcopy(config)
        pilot_config["evaluation"]["targets"] = list(TARGET_COLUMNS[:args.targets])
        if args.targets < 1 or args.targets > len(TARGET_COLUMNS):
            raise ValueError("Pilot targets must be between 1 and 17")
        if args.output is None:
            pilot_config["paths"]["results_dir"] = str(Path(config["paths"]["results_dir"]) / "pilot")
            pilot_path = resolve(root, pilot_config["paths"]["results_dir"]) / "timings.json"
        else:
            pilot_path = resolve(root, args.output)
            pilot_config["paths"]["results_dir"] = str(pilot_path.parent)
        if any(name in {"hvg_genept", "hvg_uce", "hvg_nicheformer"} for name in args.representations):
            # Validate every fold before creating the common pilot fingerprint.
            # Arrays may be queued with dependencies, but no pilot may start
            # while any declared reusable encoder artifact is incomplete.
            from .hvg_embeddings import check_hvg_embeddings
            check_hvg_embeddings(config, root)
        signature = compute_fingerprint(config, root)
        result = {"runs": [], "fingerprint": signature, "pilot_targets": args.targets}
        if pilot_path.is_file():
            previous = json.loads(pilot_path.read_text(encoding="utf-8"))
            if previous.get("fingerprint") != signature:
                raise ValueError(f"Pilot inputs/code changed; archive {pilot_path.parent} before measuring again")
            if previous.get("pilot_targets") != args.targets:
                raise ValueError("Pilot target count changed; use the original --targets value")
            result = previous
        for predictor in args.predictors:
            for representation in args.representations:
                if any(r["predictor"] == predictor and r["representation"] == representation
                       for r in result["runs"]):
                    continue
                if representation in {"hvg_genept", "hvg_uce", "hvg_nicheformer"}:
                    # Production array tasks start in separate processes. Clear
                    # validation's resource hash cache so every measured pair
                    # includes the same cold artifact validation cost.
                    from . import hvg_embeddings as hvg_artifacts
                    with hvg_artifacts._HASH_LOCK:
                        hvg_artifacts._HASH_CACHE.clear()
                started = time.perf_counter()
                measured = evaluate_job(pilot_config, root, protocol="cell_kfold", representation=representation,
                                        predictor=predictor, fold=0, device=args.device)
                # A crash before timings.json can leave complete target files. Retain
                # their original compute time instead of timing only a resumed read.
                outcomes = measured.get("targets", [])
                if len(outcomes) != args.targets or any(row.get("status") != "complete" for row in outcomes):
                    raise ValueError("Pilot did not complete every requested target")
                elapsed = max(time.perf_counter() - started,
                              sum(float(row["elapsed_seconds"]) for row in outcomes))
                result["runs"].append({"predictor": predictor, "representation": representation,
                    "targets": args.targets, "elapsed_seconds": elapsed,
                    "device": args.device,
                    "timing_method": "max(wall_time, saved_target_compute_time)"})
                atomic_json(pilot_path, result)
    elif command == "budget":
        from .jobs import estimate_jobs
        with resolve(root, args.jobs).open(newline="", encoding="utf-8") as handle:
            jobs = list(csv.DictReader(handle, delimiter="\t"))
        result = estimate_jobs(config, jobs, resolve(root, args.pilot), root=root)
        if args.require_within_budget and not result["within_budget"]:
            print(json.dumps(result, indent=2))
            raise RuntimeError("Set the actual remaining GPU hours and complete all pilots before submission")
    elif command == "historical":
        from .historical import run_historical
        result = run_historical(config, root=root, backend=args.backend)
    elif command == "report":
        from .reporting import report
        result = report(config, root=root)
    elif command == "preflight":
        import importlib.metadata
        result = {"python": sys.version, "platform": platform.platform(), "root": str(root),
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "account": config["cluster"]["account"]}
        result["versions"] = {}
        for distribution in ("numpy", "pandas", "scikit-learn", "torch", "tabpfn", "structured-data-models"):
            try:
                result["versions"][distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                result["versions"][distribution] = None
        try:
            result["gpus"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                         "--format=csv,noheader"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            result["gpus"] = None
        atomic_json(resolve(root, config["paths"]["results_dir"]) / "preflight.json", result)
    else:
        raise AssertionError(command)
    import dataclasses
    if dataclasses.is_dataclass(result):
        result = dataclasses.asdict(result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
