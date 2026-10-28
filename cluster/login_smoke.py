#!/usr/bin/env python3
"""Small GPU-login prediction diagnostic, isolated from queued benchmark pilots."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

# Reuse the tested CPU, offline environment and whole-process-group controls.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import login_hvg as controls

PREDICTORS = ("tabpfn", "limix", "kumo")
REPRESENTATIONS = ("hvg_pca128", "genept")
TARGETS = 2
TOTAL_SECONDS = 600

# Runs inside each existing native model environment. Memory figures describe
# the PyTorch allocator, not total process VRAM or the other users on a GPU.
NATIVE_RUNNER = r'''
import json, os, runpy, time
from pathlib import Path
started = time.perf_counter()
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, "Select exactly one available GPU UUID"
torch.cuda.init()
torch.cuda.reset_peak_memory_stats()
try:
    runpy.run_module("cell_property_prediction.cli", run_name="__main__")
except SystemExit as error:
    if error.code not in (None, 0):
        raise
torch.cuda.synchronize()
free, total = torch.cuda.mem_get_info()
report = {
    "wall_seconds": time.perf_counter() - started,
    "device_name": torch.cuda.get_device_name(0),
    "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
    "peak_torch_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
    "peak_torch_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    "free_device_mib_after": free / 1024**2, "total_device_mib": total / 1024**2,
}
Path(os.environ["CPP_SMOKE_RUNTIME_JSON"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
'''


def _now():
    return datetime.now(timezone.utc).isoformat()


def _selection(predictors):
    selected = list(predictors)
    if not selected or len(set(selected)) != len(selected) or any(name not in PREDICTORS for name in selected):
        raise ValueError("Select each of tabpfn, limix, kumo at most once")
    return selected


def _timings(path, predictor):
    result = json.loads(path.read_text(encoding="utf-8"))
    records = result.get("runs", [])
    if (result.get("pilot_targets") != TARGETS or len(records) != len(REPRESENTATIONS) or
            {(row.get("predictor"), row.get("representation")) for row in records} !=
            {(predictor, representation) for representation in REPRESENTATIONS} or
            any(row.get("targets") != TARGETS for row in records)):
        raise RuntimeError(f"Diagnostic did not complete the expected four target evaluations: {path}")
    return records


def run(root, config_path, gpu=None, *, execute=False, predictors=PREDICTORS):
    # Limit threads before importing any scientific libraries in the supervisor.
    os.environ.update({key: "4" for key in controls.THREAD_VARS})
    cpus = controls._bound_cpus() if execute or hasattr(os, "sched_getaffinity") else []
    from cell_property_prediction.config import atomic_json, file_sha256, load_config, resolve

    root = Path(root).resolve()
    config_path = resolve(root, config_path).resolve()
    config = load_config(config_path)
    predictors = _selection(predictors)
    base = root / "diagnostics/login_smoke"
    production = resolve(root, config["paths"].get("results_dir", "results")).resolve()
    if base.resolve().is_relative_to(production) or production.is_relative_to(base.resolve()):
        raise RuntimeError("Diagnostic outputs must be outside the production results directory")
    plan = {"predictors": predictors, "representations": list(REPRESENTATIONS),
            "targets_per_representation": TARGETS, "target_evaluations": 4 * len(predictors),
            "total_timeout_seconds": TOTAL_SECONDS, "cpu_limit": 4, "slurm_jobs_unchanged": True,
            "outputs_under": str(base), "use_for_production_budget": False}
    if not execute:
        print(json.dumps(dict(plan, execute=False), indent=2), flush=True)
        return plan
    if socket.gethostname().split(".")[0] != "login23-g-1":
        raise RuntimeError("Run this diagnostic on login23-g-1, not a regular login node")
    if not gpu or not re.fullmatch(r"(?:GPU-|MIG-)[A-Za-z0-9/_.-]+", gpu):
        raise RuntimeError("Pass an available GPU or MIG UUID from nvidia-smi -L using --gpu")
    for name in predictors:
        python = root / ".venvs" / name / "bin/python"
        if not python.is_file():
            raise RuntimeError(f"Missing existing native environment: {python}")
    state_path = base / "latest.json"
    # The same manual-GPU lock also prevents overlap with login_hvg.py.
    with controls._lock(root / "results/pending_workflow/login-hvg.lock"):
        for previous_path in (state_path, root / "results/pending_workflow/login-hvg.json"):
            if previous_path.is_file():
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
                if previous.get("root") != str(root):
                    raise RuntimeError("Previous manual GPU run belongs to another project root")
                controls._no_previous_process(previous)
        run_dir = base / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
        run_dir.mkdir(parents=True, exist_ok=False)
        state = dict(plan, schema_version=1, kind="prediction_login_smoke", root=str(root),
                     started_at=_now(), host=socket.gethostname(), gpu=gpu, cpus=cpus,
                     config_sha256=file_sha256(config_path), run_dir=str(run_dir), runs=[], status="running")

        def save():
            atomic_json(run_dir / "summary.json", state)
            atomic_json(state_path, state)

        def remember(pid):
            state["current"]["process_group"] = pid
            state["current"]["host"] = socket.gethostname()
            save()

        save()
        deadline = time.monotonic() + TOTAL_SECONDS
        try:
            for predictor in predictors:
                if predictor == "limix":
                    if time.monotonic() >= deadline:
                        raise RuntimeError("The ten-minute diagnostic budget is exhausted")
                    from limix_cache import ensure as ensure_limix_cache
                    ensure_limix_cache(config, root, apply=True)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("The ten-minute diagnostic budget is exhausted")
                directory = run_dir / predictor
                directory.mkdir()
                timings = directory / "timings.json"
                runtime = directory / "runtime.json"
                log = directory / "output.log"
                env = controls._child_env(root, gpu)
                env["CPP_SMOKE_RUNTIME_JSON"] = str(runtime)
                command = [str(root / ".venvs" / predictor / "bin/python"), "-c", NATIVE_RUNNER,
                           "--root", str(root), "--config", str(config_path), "pilot",
                           "--predictors", predictor, "--representations", *REPRESENTATIONS,
                           "--targets", str(TARGETS), "--device", "cuda", "--output", str(timings)]
                state["current"] = {"predictor": predictor, "log": str(log)}
                save()
                print(f"Testing {predictor}: two targets x two representations. Log: {log}", flush=True)
                controls._run_command(command, env, root, log, remaining, on_start=remember)
                records = _timings(timings, predictor)
                hardware = json.loads(runtime.read_text(encoding="utf-8"))
                state["runs"].append({"predictor": predictor, "pairs": records, "runtime": hardware,
                                      "timings_path": str(timings), "log": str(log)})
                state.pop("current", None)
                save()
                print(f"{predictor}: {hardware['wall_seconds']:.1f}s; peak PyTorch reserved memory "
                      f"{hardware['peak_torch_reserved_mib']:.0f} MiB", flush=True)
            state.update(status="complete", finished_at=_now())
            save()
        except BaseException as error:
            state.update(status="stopped", finished_at=_now(), error=f"{type(error).__name__}: {error}")
            save()
            print(f"Diagnostic stopped. See {state_path}. Queued Slurm jobs were unchanged.", file=sys.stderr)
            raise
    print(f"Diagnostic complete. Summary: {state_path}\n"
          "These measurements are separate from the full seventeen-target budget pilots.", flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=os.environ.get("CPP_ROOT", "."))
    parser.add_argument("--config", default=os.environ.get("CPP_CONFIG", "configs/cluster.local.yaml"))
    parser.add_argument("--gpu", help="Available GPU/MIG UUID from nvidia-smi -L")
    parser.add_argument("--execute", action="store_true", help="Start the bounded diagnostic; default is a preview")
    parser.add_argument("--predictors", nargs="+", choices=PREDICTORS, default=list(PREDICTORS))
    args = parser.parse_args()
    if args.execute:
        def stop(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")
        for name in ("SIGTERM", "SIGHUP"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), stop)
    try:
        run(args.root, args.config, args.gpu, execute=args.execute, predictors=args.predictors)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"{error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Native process group stopped; queued Slurm jobs were unchanged.\n")


if __name__ == "__main__":
    main()
