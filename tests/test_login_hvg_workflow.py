"""Mocked Slurm handover and recovery; no model downloads or GPU work."""
from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from cell_property_prediction import config as config_module
from cell_property_prediction import hvg_embeddings as native

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("login_hvg_workflow", REPO / "cluster/login_hvg.py")
login = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(login)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    if os.name == "nt":
        # The helper targets Linux; Windows antivirus can temporarily lock an
        # immediately replaced journal. Linux CI exercises the real atomic IO.
        def journal_write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        monkeypatch.setattr(config_module, "atomic_json", journal_write)
    config = {"schema_version": 1, "paths": {}, "data": {}, "models": {}, "embeddings": {},
              "evaluation": {"protocols": ["cell_kfold", "mouse_group_kfold"], "n_splits": 10},
              "cluster": {"account": "thes2301"}}
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    directory = root / "results/pending_workflow"
    directory.mkdir(parents=True)
    ids = directory / "slurm-jobs.tsv"
    roles = list(login.ROLES)
    ids.write_text("".join(f"{role}\t{100 + n}\n" for n, role in enumerate(roles)))
    queue = [{"job_id": str(100+n), "state": "PENDING", "workdir": str(root),
              "name": login.ROLES[role], "user": "tester", "account": "thes2301",
              "reason": "Priority" if n < 2 else "Dependency"} for n, role in enumerate(roles)]
    events = []
    tasks = [{"encoder": "uce", "protocol": "cell_kfold", "fold": 1},
             {"encoder": "nicheformer", "protocol": "cell_kfold", "fold": 1}]
    for encoder in login.ENCODERS:
        path = root / ".venvs" / encoder / "bin/python"
        path.parent.mkdir(parents=True)
        path.touch()
    monkeypatch.setattr(login, "_lock", lambda path: nullcontext())
    monkeypatch.setattr(login, "_bound_cpus", lambda: [2, 4, 6, 8])
    monkeypatch.setattr(login.socket, "gethostname", lambda: "login23-g-1.example")
    monkeypatch.setattr(login.getpass, "getuser", lambda: "tester")
    monkeypatch.setattr(login, "_queue", lambda: [dict(row) for row in queue])
    monkeypatch.setattr(login, "_missing", lambda config, root: list(tasks))

    def control(*args):
        events.append(tuple(args))
        if args[:2] == ("scontrol", "hold"):
            for row in queue:
                if row["job_id"] == args[2]:
                    row["reason"] = "JobHeldUser"
        elif args[:2] == ("scontrol", "release"):
            for row in queue:
                if row["job_id"] == args[2]:
                    row["reason"] = "Priority"
        elif args[0] == "scancel":
            queue[:] = [row for row in queue if row["job_id"] not in args[1:]]
        else:
            raise AssertionError(args)

    monkeypatch.setattr(login, "_control", control)

    def paths(config, root, name, protocol, fold):
        return root / f"{name}-{protocol}-{fold}.npz", root / f"{name}-{protocol}-{fold}.audit.json"

    monkeypatch.setattr(native, "_embedding_paths", paths)
    monkeypatch.setattr(native, "_validate_embedding", lambda *args, **kwargs: events.append(("validate", args[2])))

    def check(config, root):
        events.append(("check_all",))
        return {"n_embeddings": 60, "valid": True}

    monkeypatch.setattr(native, "check_hvg_embeddings", check)

    def command(command, env, root, log, timeout, on_start=None):
        events.append(("native", command, timeout, dict(env)))
        if "embed" in command:
            name = command[command.index("--encoder") + 1]
            protocol = command[command.index("--protocol") + 1]
            fold = int(command[command.index("--fold") + 1])
            paths(config, root, name, protocol, fold)[0].write_bytes(b"synthetic fixture")
            tasks[:] = [task for task in tasks if task["encoder"] != name]

    monkeypatch.setattr(login, "_run_command", command)
    return SimpleNamespace(root=root, config_path=config_path, queue=queue, tasks=tasks, events=events,
                           state=directory / "login-hvg.json", ids=ids, command=command)


def execute(rig, **kwargs):
    return login.run(rig.root, rig.config_path, "GPU-test-device", execute=True, **kwargs)


def test_dry_run_does_not_hold_or_compute(rig):
    result = login.run(rig.root, rig.config_path, None)
    assert result["missing_folds"] == 2
    assert len(result["jobs_to_hold"]) == 5
    assert rig.events == []
    assert not rig.state.exists()


def test_full_handover_validates_before_cancel_preserves_history_and_config(rig):
    before = rig.ids.read_bytes(), rig.config_path.read_bytes()
    result = execute(rig)
    assert result["complete"] and result["validated_embeddings"] == 60
    assert rig.events.index(("check_all",)) < next(i for i, event in enumerate(rig.events) if event[0] == "scancel")
    assert rig.ids.read_bytes() == before[0] and rig.config_path.read_bytes() == before[1]
    native_calls = [event for event in rig.events if event[0] == "native" and "embed" in event[1]]
    assert len(native_calls) == 2
    assert all(event[1][-2:] == ["--device", "cuda"] and event[2] == 600 for event in native_calls)
    assert all(event[3]["CUDA_VISIBLE_DEVICES"] == "GPU-test-device" for event in native_calls)
    assert not rig.queue
    state = json.loads(rig.state.read_text())
    assert state["status"] == "complete" and len(state["completed"]) == 2
    assert not any(event[0] == "sbatch" for event in rig.events)


def test_running_encoder_is_held_but_never_duplicated(rig):
    rig.queue[0]["state"] = "RUNNING"
    with pytest.raises(RuntimeError, match="running or not held"):
        execute(rig)
    assert not any(event[0] == "native" for event in rig.events)
    assert all(row["reason"] == "JobHeldUser" for row in rig.queue)
    rig.queue[0]["state"] = "PENDING"
    assert execute(rig)["complete"]


def test_running_predictor_prevents_any_mutation(rig):
    rig.queue[2]["state"] = "RUNNING"
    with pytest.raises(RuntimeError, match="predictor pilot"):
        execute(rig)
    assert not rig.events


@pytest.mark.parametrize("field,value", [("account", "someone_else"), ("name", "other"), ("user", "other")])
def test_mismatched_job_is_never_held(rig, field, value):
    rig.queue[0][field] = value
    with pytest.raises(RuntimeError, match="ownership"):
        execute(rig)
    assert not rig.events


def test_partial_trial_releases_only_acquired_holds(rig):
    rig.queue[2]["reason"] = "JobHeldUser"
    result = execute(rig, max_folds=1)
    assert result == {"complete": False, "completed_now": 1}
    assert not any(event[0] in {"scancel", "check_all"} for event in rig.events)
    released = [event[2] for event in rig.events if event[:2] == ("scontrol", "release")]
    assert set(released) == {"100", "101", "103", "104"}
    assert rig.queue[2]["reason"] == "JobHeldUser"


def test_failed_native_preserves_holds_and_never_cancels(rig, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(login, "_run_command", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        execute(rig)
    assert all(row["reason"] == "JobHeldUser" for row in rig.queue)
    assert not any(event[0] == "scancel" for event in rig.events)
    assert json.loads(rig.state.read_text())["status"] == "stopped"


def test_all_sixty_validation_failure_never_cancels(rig, monkeypatch):
    monkeypatch.setattr(native, "check_hvg_embeddings", lambda *args: {"n_embeddings": 59, "valid": True})
    with pytest.raises(RuntimeError, match="sixty"):
        execute(rig)
    assert not any(event[0] == "scancel" for event in rig.events)


@pytest.mark.parametrize("action", ["resume", "release"])
def test_orphaned_native_group_blocks_resume_and_release(rig, monkeypatch, action):
    rig.state.write_text(json.dumps({"schema_version": 1, "root": str(rig.root), "held_by_runner": ["100"],
                                   "current": {"host": "login23-g-1.example", "process_group": 456}}))
    monkeypatch.setattr(login.os, "killpg", lambda *args: None, raising=False)
    with pytest.raises(RuntimeError, match="still active"):
        if action == "resume":
            execute(rig)
        else:
            login.release(rig.root, rig.config_path)
    assert not rig.events


def test_wrong_host_and_missing_uuid_stop_before_holds(rig, monkeypatch):
    monkeypatch.setattr(login.socket, "gethostname", lambda: "login23-1")
    with pytest.raises(RuntimeError, match="Execute only"):
        execute(rig)
    assert not rig.events
    monkeypatch.setattr(login.socket, "gethostname", lambda: "login23-g-1")
    with pytest.raises(RuntimeError, match="UUID"):
        login.run(rig.root, rig.config_path, None, execute=True)
    assert not rig.events


@pytest.mark.parametrize("action", ["resume", "release"])
def test_orphaned_prediction_diagnostic_blocks_hvg_helper(rig, monkeypatch, action):
    path = rig.root / "diagnostics/login_smoke/latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"root": str(rig.root), "current": {
        "process_group": 456, "host": "login23-g-1.example"}}))
    monkeypatch.setattr(login.os, "killpg", lambda *args: None, raising=False)
    with pytest.raises(RuntimeError, match="still active"):
        if action == "resume":
            execute(rig)
        else:
            login.release(rig.root, rig.config_path)
    assert not rig.events
