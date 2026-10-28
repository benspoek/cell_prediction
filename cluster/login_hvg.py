#!/usr/bin/env python3
"""Run missing HVG encoders serially on the RWTH GPU login node.

Operational helper only: scientific sources, configuration and existing artifacts
are never modified. Use the existing core environment to start this supervisor.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import uuid

ENCODERS = ("uce", "nicheformer")
ROLES = {"hvg_encoder_uce": "cellhvgencoder",
         "hvg_encoder_nicheformer": "cellhvgencoder",
         "pilot_tabpfn": "cellpilot", "pilot_limix": "cellpilot", "pilot_kumo": "cellpilot"}
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "OMP_THREAD_LIMIT")


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _child_env(root: Path, gpu: str) -> dict:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("SLURM_") or key in {
                "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "NODE_RANK", "CUDA_DEVICE_ORDER"}:
            env.pop(key, None)
    env.update({key: "4" for key in THREAD_VARS})
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
               HF_HOME=str(root / "cache/huggingface"),
               HF_HUB_CACHE=str(root / "cache/huggingface/hub"),
               TRANSFORMERS_CACHE=str(root / "cache/huggingface/hub"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    return env


def _bound_cpus() -> list[int]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("Linux CPU affinity is required to enforce the four-CPU login limit")
    cpus = sorted(os.sched_getaffinity(0))[:4]
    if not cpus:
        raise RuntimeError("No allowed CPUs")
    os.sched_setaffinity(0, cpus)
    return cpus


def _queue() -> list[dict]:
    result = subprocess.run(["squeue", "-h", "--me", "-o", "%F|%T|%Z|%j|%u|%a|%R"],
                            check=True, capture_output=True, text=True, timeout=30)
    rows = []
    keys = ("job_id", "state", "workdir", "name", "user", "account", "reason")
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split("|", 6)]
        if len(fields) != 7 or not fields[0].isdigit():
            raise RuntimeError(f"Malformed squeue response: {line!r}")
        rows.append(dict(zip(keys, fields)))
    return rows


def _stop_group(process):
    # The parent can exit before an orphaned vendor/accelerate child does.
    # Always signal the whole group, including after the parent has exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def _run_command(command: list[str], env: dict, root: Path, log: Path, timeout: int, on_start=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if on_start is not None:
                on_start(process.pid)
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            _stop_group(process)
            raise RuntimeError(f"Encoder timed out after {timeout}s; inspect {log}. "
                               "Partial outputs were preserved.") from error
        except BaseException:
            _stop_group(process)
            raise
    if code:
        raise RuntimeError(f"Encoder exited with status {code}; inspect {log}. "
                           "Partial outputs were preserved.")


@contextlib.contextmanager
def _lock(path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another login-HVG supervisor is active") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _saved_jobs(root):
    path = root / "results/pending_workflow/slurm-jobs.tsv"
    jobs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2 or not fields[1].isdigit():
            raise RuntimeError(f"Invalid saved job record: {line!r}")
        role, job_id = fields
        if job_id in jobs and jobs[job_id] != role:
            raise RuntimeError(f"Conflicting roles for saved job {job_id}")
        jobs[job_id] = role
    return jobs


def _relevant_queue(root, account, saved):
    rows = []
    for row in _queue():
        job_id, role = row["job_id"], saved.get(row["job_id"])
        same_root = Path(row["workdir"]).resolve() == root
        if role in ROLES:
            if (not same_root or row["user"] != getpass.getuser() or
                    row["account"] != account or row["name"] != ROLES[role]):
                raise RuntimeError(f"Saved job {job_id} ownership, root, account or name differs")
            row = dict(row, role=role)
            rows.append(row)
        elif same_root and row["name"].startswith("cell"):
            raise RuntimeError(f"Other project job {job_id} ({row['name']}) is active; "
                               "finish it before login encoding")
    return rows


def _missing(config, root):
    from cell_property_prediction import hvg_embeddings as native

    if (list(config["evaluation"]["protocols"]) != ["cell_kfold", "mouse_group_kfold"] or
            int(config["evaluation"]["n_splits"]) != 10):
        raise RuntimeError("This continuation requires the existing two ten-fold protocols")
    tasks = []
    for protocol in config["evaluation"]["protocols"]:
        if len(native._plan(config, root, protocol)["folds"]) != 10:
            raise RuntimeError("Expected ten saved folds per protocol")
        for fold in range(10):
            for encoder in ENCODERS:
                output, audit = native._embedding_paths(config, root, encoder, protocol, fold)
                if output.exists() or audit.exists():
                    native._validate_embedding(config, root, encoder, protocol, fold, runtime=False)
                else:
                    tasks.append({"encoder": encoder, "protocol": protocol, "fold": fold})
    return tasks


def _write_state(path, state):
    from cell_property_prediction.config import atomic_json

    state["updated_at"] = _stamp()
    atomic_json(path, state)


def _control(*args):
    subprocess.run(list(args), check=True, capture_output=True, text=True, timeout=30)


def _no_previous_process(state):
    previous = state.get("current", {})
    if previous.get("process_group"):
        if previous.get("host", "").split(".")[0] != socket.gethostname().split(".")[0]:
            raise RuntimeError("A previous native process was recorded on another host; inspect it first")
        try:
            os.killpg(int(previous["process_group"]), 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"Previous native process group {previous['process_group']} is still active; "
                               "do not start a duplicate. Inspect it before resuming")


def _check_smoke_process(root):
    path = root / "diagnostics/login_smoke/latest.json"
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("root") != str(root):
            raise RuntimeError("Previous prediction diagnostic belongs to another project root")
        _no_previous_process(state)


def _release(root, config, saved, state_path, state):
    _no_previous_process(state)
    rows = _relevant_queue(root, str(config["cluster"]["account"]), saved)
    by_id = {}
    for row in rows:
        by_id.setdefault(row["job_id"], []).append(row)
    for job_id in list(state.get("held_by_runner", [])):
        active = by_id.get(job_id, [])
        if active and any(row["state"] != "PENDING" for row in active):
            raise RuntimeError(f"Job {job_id} is not pending; inspect Slurm before release")
        if active:
            _control("scontrol", "release", job_id)
        state["held_by_runner"].remove(job_id)
        _write_state(state_path, state)
    state["status"] = "released"
    _write_state(state_path, state)


def release(root, config_path):
    from cell_property_prediction.config import load_config, resolve

    root = Path(root).resolve()
    config = load_config(resolve(root, config_path))
    directory = root / "results/pending_workflow"
    with _lock(directory / "login-hvg.lock"):
        _check_smoke_process(root)
        state_path = directory / "login-hvg.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("root") != str(root):
            raise RuntimeError("Login-HVG state belongs to another project root")
        _release(root, config, _saved_jobs(root), state_path, state)
    print("Released only holds recorded by this helper. Check pending.sh status.", flush=True)


def run(root, config_path, gpu, execute=False, max_folds=None):
    # Do this before NumPy/SciPy import so BLAS threads inherit the same limit.
    os.environ.update({key: "4" for key in THREAD_VARS})
    cpus = _bound_cpus() if execute or hasattr(os, "sched_getaffinity") else []
    from cell_property_prediction.config import file_sha256, load_config, resolve
    from cell_property_prediction import hvg_embeddings as native

    root = Path(root).resolve()
    config_path = resolve(root, config_path).resolve()
    config = load_config(config_path)
    saved = _saved_jobs(root)
    if max_folds is not None and (isinstance(max_folds, bool) or max_folds < 1):
        raise ValueError("max_folds must be a positive integer")
    if execute and socket.gethostname().split(".")[0] != "login23-g-1":
        raise RuntimeError("Execute only on login23-g-1; regular login nodes have no GPU")
    if execute and (not gpu or not re.fullmatch(r"(?:GPU-|MIG-)[A-Za-z0-9/_.-]+", gpu)):
        raise RuntimeError("Pass the available GPU or MIG UUID from nvidia-smi -L using --gpu")
    directory = root / "results/pending_workflow"
    state_path = directory / "login-hvg.json"
    with _lock(directory / "login-hvg.lock"):
        _check_smoke_process(root)
        rows = _relevant_queue(root, str(config["cluster"]["account"]), saved)
        if not execute:
            tasks = _missing(config, root)
            result = {"execute": False, "missing_folds": len(tasks), "tasks": tasks,
                      "jobs_to_hold": sorted({row["job_id"] for row in rows}),
                      "timeout_seconds_per_fold": 600, "cpu_limit": 4}
            print(json.dumps(result, indent=2), flush=True)
            return result
        # Reuse our journal after a timeout, interrupted session or partial run.
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
            "schema_version": 1, "root": str(root), "held_by_runner": [], "completed": []}
        if state.get("root") != str(root) or state.get("schema_version") != 1:
            raise RuntimeError("Unexpected login-HVG recovery state")
        _no_previous_process(state)
        if any(row["role"].startswith("pilot_") and row["state"] != "PENDING" for row in rows):
            raise RuntimeError("A predictor pilot is already running; do not duplicate its dependencies")
        state.update(status="holding", gpu=gpu, cpus=cpus, host=socket.gethostname(),
                     config_sha256=file_sha256(config_path), started_at=_stamp())
        _write_state(state_path, state)
        try:
            # Holding a parent prevents future array tasks, but does not stop a
            # task that has already started. Recheck after all hold operations.
            grouped = {}
            for row in rows:
                grouped.setdefault(row["job_id"], []).append(row)
            for job_id, entries in grouped.items():
                if all(row["reason"] == "JobHeldUser" for row in entries):
                    continue
                if job_id not in state["held_by_runner"]:
                    state["held_by_runner"].append(job_id)
                    _write_state(state_path, state)
                _control("scontrol", "hold", job_id)
            rows = _relevant_queue(root, str(config["cluster"]["account"]), saved)
            if any(row["state"] != "PENDING" or row["reason"] != "JobHeldUser" for row in rows):
                raise RuntimeError("Some saved tasks are running or not held. Let running tasks finish, "
                                   "then rerun this helper; future tasks are held")
            tasks = _missing(config, root)
            selected = tasks[:max_folds] if max_folds else tasks
            state.update(status="encoding", missing_at_start=len(tasks))
            _write_state(state_path, state)
            env = _child_env(root, gpu)
            # Verify the explicit visible device in a short-lived native process;
            # this creates no persistent CUDA context in the supervisor.
            def remember_process(pid):
                state["current"]["process_group"] = pid
                state["current"]["host"] = socket.gethostname()
                _write_state(state_path, state)

            for encoder in sorted({task["encoder"] for task in selected}):
                python = root / ".venvs" / encoder / "bin/python"
                if not python.is_file():
                    raise RuntimeError(f"Missing existing encoder interpreter {python}")
                probe = ("import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
                         "assert torch.cuda.device_count() == 1, 'Expected one visible GPU'; "
                         "print(torch.cuda.get_device_name(0)); print(torch.cuda.mem_get_info())")
                probe_log = directory / "login-hvg-logs" / f"probe-{encoder}-{uuid.uuid4().hex}.log"
                state["current"] = {"probe": encoder, "log": str(probe_log)}
                _write_state(state_path, state)
                _run_command([str(python), "-c", probe], env, root, probe_log, 60,
                             on_start=remember_process)
                state.pop("current", None)
                _write_state(state_path, state)
            for number, task in enumerate(selected, 1):
                name, protocol, fold = task["encoder"], task["protocol"], task["fold"]
                output, audit = native._embedding_paths(config, root, name, protocol, fold)
                if output.exists() or audit.exists():
                    raise RuntimeError(f"An unexpected writer created {output}; inspect before resuming")
                log = directory / "login-hvg-logs" / f"{name}-{protocol}-{fold:02d}-{uuid.uuid4().hex}.log"
                command = [str(root / ".venvs" / name / "bin/python"), "-m",
                           "cell_property_prediction.cli", "--root", str(root), "--config", str(config_path),
                           "embed", "--gene-set", "hvg", "--encoder", name,
                           "--protocol", protocol, "--fold", str(fold), "--device", "cuda"]
                state["current"] = dict(task, log=str(log))
                _write_state(state_path, state)
                print(f"[{number}/{len(selected)}] {name} {protocol} fold {fold}; log={log}", flush=True)
                _run_command(command, env, root, log, 600, on_start=remember_process)
                native._validate_embedding(config, root, name, protocol, fold, runtime=False)
                state["completed"].append(dict(task, log=str(log), finished_at=_stamp(),
                                               output_sha256=file_sha256(output)))
                state.pop("current", None)
                _write_state(state_path, state)
            if len(selected) < len(tasks):
                _release(root, config, saved, state_path, state)
                print("Selected folds completed. Batch holds released; remaining tasks stay in Slurm.", flush=True)
                return {"complete": False, "completed_now": len(selected)}
            checked = native.check_hvg_embeddings(config, root)
            if checked["n_embeddings"] != 60 or checked.get("valid") is not True:
                raise RuntimeError("Expected all sixty validated HVG embeddings")
            state.update(status="validated", validated_embeddings=60)
            _write_state(state_path, state)
            rows = _relevant_queue(root, str(config["cluster"]["account"]), saved)
            if any(row["state"] != "PENDING" or row["reason"] != "JobHeldUser" for row in rows):
                raise RuntimeError("A replacement job is no longer held; inspect before canceling")
            ids = sorted({row["job_id"] for row in rows})
            if ids:
                _control("scancel", *ids)
                if _relevant_queue(root, str(config["cluster"]["account"]), saved):
                    raise RuntimeError("Canceled jobs still appear in Slurm. Wait briefly and rerun "
                                       "this helper to confirm completion")
            state.update(status="complete", held_by_runner=[], canceled_jobs=ids, finished_at=_stamp())
            _write_state(state_path, state)
            print("All 60 HVG embeddings validate. Old held arrays and pending pilots canceled.\n"
                  "Next: zsh cluster/pending.sh pilot", flush=True)
            return {"complete": True, "validated_embeddings": 60, "canceled_jobs": ids}
        except BaseException as error:
            state.update(status="stopped", error=f"{type(error).__name__}: {error}")
            _write_state(state_path, state)
            print("Stopped. Existing artifacts and job history were preserved. Holds remain.\n"
                  "Resume with this helper, or use --release to return held jobs to Slurm.", file=sys.stderr)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=os.environ.get("CPP_ROOT", "."))
    parser.add_argument("--config", default=os.environ.get("CPP_CONFIG", "configs/cluster.local.yaml"))
    parser.add_argument("--gpu", help="Available GPU/MIG UUID from nvidia-smi -L")
    parser.add_argument("--execute", action="store_true", help="Hold queued jobs and compute missing folds")
    parser.add_argument("--max-folds", type=int, help="Run only N missing folds, then release acquired holds")
    parser.add_argument("--release", action="store_true", help="Release only holds recorded by this helper")
    args = parser.parse_args()
    if args.release and (args.execute or args.max_folds is not None):
        parser.error("--release cannot be combined with --execute or --max-folds")
    if args.execute:
        def stop_on_signal(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")
        for name in ("SIGTERM", "SIGHUP"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), stop_on_signal)
    try:
        if args.release:
            release(args.root, args.config)
        else:
            run(args.root, args.config, args.gpu, args.execute, args.max_folds)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"{error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; native process group stopped. Check login-hvg.json and holds.\n")


if __name__ == "__main__":
    main()
