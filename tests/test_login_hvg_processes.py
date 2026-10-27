"""Portable process-control tests for the Linux-only login encoder supervisor."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cell_property_prediction import hvg_embeddings as native

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("login_hvg_processes", ROOT / "cluster/login_hvg.py")
login = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(login)


def test_children_receive_offline_single_gpu_and_four_threads(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("RANK", "8")
    monkeypatch.setenv("WORLD_SIZE", "9")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("MASTER_ADDR", "elsewhere")
    monkeypatch.setenv("OMP_NUM_THREADS", "12")
    result = login._child_env(tmp_path, "GPU-specific-uuid")
    assert result["CUDA_VISIBLE_DEVICES"] == "GPU-specific-uuid"
    assert all(result[key] == "4" for key in login.THREAD_VARS)
    assert result["HF_HUB_OFFLINE"] == result["TRANSFORMERS_OFFLINE"] == "1"
    assert result["HF_HOME"] == str(tmp_path / "cache/huggingface")
    assert not any(key.startswith("SLURM_") for key in result)
    assert not set(result) & {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR"}
    assert os.environ["SLURM_CPUS_PER_TASK"] == "16"


def test_cpu_affinity_uses_at_most_four_allowed_cpus(monkeypatch):
    calls = []
    monkeypatch.setattr(login.os, "sched_getaffinity", lambda pid: {11, 9, 7, 5, 3, 1}, raising=False)
    monkeypatch.setattr(login.os, "sched_setaffinity", lambda pid, cpus: calls.append((pid, list(cpus))), raising=False)
    assert login._bound_cpus() == [1, 3, 5, 7]
    assert calls == [(0, [1, 3, 5, 7])]


def test_queue_records_keep_identity_and_workdir(monkeypatch):
    def command(args, **kwargs):
        assert args[-1] == "%F|%T|%Z|%j|%u|%a|%R" and kwargs["timeout"] == 30
        return SimpleNamespace(stdout="5253925|PENDING|/home/u/repo|cellhvgencoder|u|thes2301|Priority\n")
    monkeypatch.setattr(login.subprocess, "run", command)
    assert login._queue() == [{"job_id": "5253925", "state": "PENDING", "workdir": "/home/u/repo",
                              "name": "cellhvgencoder", "user": "u", "account": "thes2301", "reason": "Priority"}]


@pytest.mark.parametrize("row", ["broken", "xyz|PENDING|/tmp|x|u|a|None"])
def test_malformed_queue_fails_closed(monkeypatch, row):
    monkeypatch.setattr(login.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=row))
    with pytest.raises(RuntimeError, match="Malformed"):
        login._queue()


@pytest.mark.parametrize("outcome", [0, 7, "timeout", "interrupt", "callback_error"])
def test_process_group_cleanup_and_exit_handling(tmp_path, monkeypatch, outcome):
    waits, signals, options = [], [], {}
    monkeypatch.setattr(login.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(login.os, "killpg", lambda pid, sig: signals.append((pid, sig)), raising=False)

    class Process:
        pid = 1234
        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                if outcome == "timeout":
                    raise subprocess.TimeoutExpired(["encoder"], timeout)
                if outcome == "interrupt":
                    raise KeyboardInterrupt()
                return outcome if isinstance(outcome, int) else 0
            return 0

    def popen(args, **kwargs):
        options.update(kwargs)
        return Process()

    def started(pid):
        assert pid == 1234
        if outcome == "callback_error":
            raise RuntimeError("journal failure")

    monkeypatch.setattr(login.subprocess, "Popen", popen)
    args = (["encoder"], {"CUSTOM": "1"}, tmp_path, tmp_path / "logs/fold.log", 600)
    if outcome == 0:
        assert login._run_command(*args, on_start=started) is None
    elif outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            login._run_command(*args, on_start=started)
    else:
        with pytest.raises(RuntimeError):
            login._run_command(*args, on_start=started)
    assert options["start_new_session"] is True and options["stderr"] == subprocess.STDOUT
    assert options["env"] == {"CUSTOM": "1"} and options["cwd"] == tmp_path
    if outcome in {"timeout", "interrupt", "callback_error"}:
        # SIGKILL still reaches children if the parent exits immediately on TERM.
        assert signals == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]
    else:
        assert signals == []


def test_unresponsive_group_is_force_killed(monkeypatch):
    calls = []
    monkeypatch.setattr(login.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(login.os, "killpg", lambda pid, sig: calls.append(sig), raising=False)
    class Process:
        pid = 9
        count = 0
        def wait(self, timeout):
            self.count += 1
            if self.count == 1:
                raise subprocess.TimeoutExpired(["encoder"], timeout)
            return 0
    login._stop_group(Process())
    assert calls == [signal.SIGTERM, signal.SIGKILL]


def test_existing_fold_is_validated_and_skipped_partial_is_not_overwritten(tmp_path, monkeypatch):
    config = {"evaluation": {"protocols": ["cell_kfold", "mouse_group_kfold"], "n_splits": 10}}
    monkeypatch.setattr(native, "_plan", lambda *args: {"folds": list(range(10))})
    def paths(config, root, name, protocol, fold):
        return root / f"{name}-{protocol}-{fold}.npz", root / f"{name}-{protocol}-{fold}.audit.json"
    monkeypatch.setattr(native, "_embedding_paths", paths)
    calls = []
    def validate(*args, **kwargs):
        calls.append((args[2:], kwargs))
        out, audit = paths(*args)
        if not audit.exists():
            raise ValueError("partial immutable artifact")
    monkeypatch.setattr(native, "_validate_embedding", validate)
    output, audit = paths(config, tmp_path, "uce", "cell_kfold", 0)
    output.write_bytes(b"existing")
    audit.write_text("{}")
    tasks = login._missing(config, tmp_path)
    assert len(tasks) == 39 and calls == [(("uce", "cell_kfold", 0), {"runtime": False})]
    audit.unlink()
    with pytest.raises(ValueError, match="partial immutable"):
        login._missing(config, tmp_path)
    assert output.read_bytes() == b"existing"


@pytest.mark.skipif(os.name != "posix", reason="Native Linux process-group behavior")
def test_real_timeout_stops_a_native_process(tmp_path):
    log = tmp_path / "timeout.log"
    with pytest.raises(RuntimeError, match="timed out"):
        login._run_command([sys.executable, "-c", "import time; time.sleep(30)"],
                           dict(os.environ), tmp_path, log, 0.1)
    assert log.exists()
