"""Small GPU diagnostic stays separate from queued pilot outputs and Slurm state."""
from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest
import yaml

from cell_property_prediction import config as config_module

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cluster"))
import limix_cache
SPEC = importlib.util.spec_from_file_location("login_smoke_test", REPO / "cluster/login_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump({"schema_version": 1, "paths": {"results_dir": "results"},
                                         "data": {}, "evaluation": {}, "models": {}, "embeddings": {}, "cluster": {}}))
    production = root / "results/pending_workflow/pilot_with_hvg_encoders/timings.json"
    production.parent.mkdir(parents=True)
    production.write_bytes(b"original queued-pilot timings")
    for name in smoke.PREDICTORS:
        python = root / ".venvs" / name / "bin/python"
        python.parent.mkdir(parents=True)
        python.touch()
    calls = []
    monkeypatch.setattr(limix_cache, "ensure", lambda *args, **kwargs: {"status": "verified"})
    monkeypatch.setattr(smoke.controls, "_bound_cpus", lambda: [0, 1, 2, 3])
    monkeypatch.setattr(smoke.controls, "_lock", lambda path: nullcontext())
    monkeypatch.setattr(smoke.socket, "gethostname", lambda: "login23-g-1.example")
    for key in smoke.controls.THREAD_VARS:
        monkeypatch.setenv(key, os.environ.get(key, "4"))
    if os.name == "nt":
        # Linux CI exercises the real atomic writer. Windows antivirus can
        # temporarily lock immediately replaced journals during rapid mocks.
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        monkeypatch.setattr(config_module, "atomic_json", write)

    def forbidden(*args, **kwargs):
        raise AssertionError("A diagnostic must not query or change Slurm jobs")
    monkeypatch.setattr(smoke.controls, "_control", forbidden)
    monkeypatch.setattr(smoke.controls, "_queue", forbidden)

    def run(command, env, cwd, log, timeout, on_start=None):
        calls.append((command, dict(env), cwd, log, timeout))
        if on_start:
            on_start(12345)
        model = command[command.index("--predictors") + 1]
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"pilot_targets": 2, "runs": [
            {"predictor": model, "representation": name, "targets": 2, "elapsed_seconds": 1.0,
             "device": "cuda"} for name in smoke.REPRESENTATIONS]}))
        Path(env["CPP_SMOKE_RUNTIME_JSON"]).write_text(json.dumps({
            "wall_seconds": 2.0, "peak_torch_reserved_mib": 1024, "peak_torch_allocated_mib": 900}))

    monkeypatch.setattr(smoke.controls, "_run_command", run)
    return SimpleNamespace(root=root, config=config_path, production=production, calls=calls,
                           latest=root / "diagnostics/login_smoke/latest.json", run=run)


def execute(rig, **kwargs):
    return smoke.run(rig.root, rig.config, "GPU-test-id", execute=True, **kwargs)


def test_preview_is_small_and_does_not_start_models_or_slurm(rig):
    plan = smoke.run(rig.root, rig.config)
    assert plan["target_evaluations"] == 12 and plan["total_timeout_seconds"] == 600
    assert plan["representations"] == ["hvg_pca128", "genept"]
    assert not plan["use_for_production_budget"] and plan["slurm_jobs_unchanged"]
    assert not rig.calls and not rig.latest.exists()


def test_three_models_are_serial_and_outputs_never_touch_production(rig):
    original_config = rig.config.read_bytes()
    result = execute(rig)
    assert result["status"] == "complete" and len(result["runs"]) == 3
    assert [call[0][call[0].index("--predictors") + 1] for call in rig.calls] == list(smoke.PREDICTORS)
    for command, env, cwd, log, timeout in rig.calls:
        assert command[1:3] == ["-c", smoke.NATIVE_RUNNER]
        assert command[command.index("--targets") + 1] == "2"
        assert command[command.index("--device") + 1] == "cuda"
        assert Path(command[command.index("--output") + 1]).is_relative_to(rig.root / "diagnostics/login_smoke")
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-test-id"
        assert env["HF_HUB_OFFLINE"] == "1" and env["OMP_NUM_THREADS"] == "4"
        assert 0 < timeout <= 600 and cwd == rig.root
    assert rig.production.read_bytes() == b"original queued-pilot timings"
    assert rig.config.read_bytes() == original_config
    assert json.loads(rig.latest.read_text())["status"] == "complete"
    assert not (rig.root / "results/pending_workflow/slurm-jobs.tsv").exists()


def test_each_model_receives_only_the_remaining_total_deadline(rig, monkeypatch):
    clock = iter([100, 101, 150, 151, 201])
    monkeypatch.setattr(smoke.time, "monotonic", lambda: next(clock))
    execute(rig)
    assert [call[4] for call in rig.calls] == [599, 549, 499]


def test_exhausted_deadline_stops_before_another_model(rig, monkeypatch):
    clock = iter([100, 101, 701])
    monkeypatch.setattr(smoke.time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="budget is exhausted"):
        execute(rig)
    assert len(rig.calls) == 1 and json.loads(rig.latest.read_text())["status"] == "stopped"
    assert rig.production.read_bytes() == b"original queued-pilot timings"


@pytest.mark.parametrize("failure", [RuntimeError("timed out"), KeyboardInterrupt()])
def test_failure_keeps_diagnostics_and_leaves_full_pilots_untouched(rig, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(smoke.controls, "_run_command", fail)
    with pytest.raises(type(failure)):
        execute(rig)
    state = json.loads(rig.latest.read_text())
    assert state["status"] == "stopped" and Path(state["run_dir"]).exists()
    assert rig.production.read_bytes() == b"original queued-pilot timings"


def test_followup_model_is_fresh_and_does_not_reuse_previous_targets(rig):
    first = execute(rig, predictors=["tabpfn"])
    second = execute(rig, predictors=["tabpfn"])
    assert first["run_dir"] != second["run_dir"] and len(rig.calls) == 2
    assert all(Path(result["run_dir"]).exists() for result in [first, second])


def test_live_previous_native_group_prevents_duplicate_test(rig, monkeypatch):
    rig.latest.parent.mkdir(parents=True)
    rig.latest.write_text(json.dumps({"root": str(rig.root), "current": {
        "process_group": 12345, "host": "login23-g-1.example"}}))
    monkeypatch.setattr(smoke.controls.os, "killpg", lambda *args: None, raising=False)
    with pytest.raises(RuntimeError, match="still active"):
        execute(rig)
    assert not rig.calls


def test_host_uuid_and_output_root_guards(rig, monkeypatch):
    monkeypatch.setattr(smoke.socket, "gethostname", lambda: "login23-1")
    with pytest.raises(RuntimeError, match="not a regular login"):
        execute(rig)
    monkeypatch.setattr(smoke.socket, "gethostname", lambda: "login23-g-1")
    with pytest.raises(RuntimeError, match="UUID"):
        smoke.run(rig.root, rig.config, None, execute=True)
    config = yaml.safe_load(rig.config.read_text())
    config["paths"]["results_dir"] = "."
    rig.config.write_text(yaml.safe_dump(config))
    with pytest.raises(RuntimeError, match="outside the production"):
        execute(rig)
    assert not rig.calls


@pytest.mark.parametrize("models", [[], ["tabpfn", "tabpfn"], ["unknown"]])
def test_bad_selection_is_rejected(rig, models):
    with pytest.raises(ValueError, match="at most once"):
        execute(rig, predictors=models)
    assert not rig.calls


def test_native_wrapper_records_allocator_peaks_without_real_torch(tmp_path, monkeypatch):
    output = tmp_path / "runtime.json"
    monkeypatch.setenv("CPP_SMOKE_RUNTIME_JSON", str(output))
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, init=lambda: None,
                           reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
                           mem_get_info=lambda: (100*1024**2, 200*1024**2),
                           get_device_name=lambda index: "Fake GPU", max_memory_allocated=lambda: 20*1024**2,
                           max_memory_reserved=lambda: 30*1024**2)
    fake = SimpleNamespace(cuda=cuda, __version__="fake", version=SimpleNamespace(cuda="fake"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    def native(*args, **kwargs):
        raise SystemExit(0)
    monkeypatch.setattr(runpy, "run_module", native)
    exec(smoke.NATIVE_RUNNER, {"__name__": "__main__"})
    result = json.loads(output.read_text())
    assert result["peak_torch_allocated_mib"] == 20 and result["peak_torch_reserved_mib"] == 30
    assert result["device_name"] == "Fake GPU" and result["total_device_mib"] == 200


def test_live_orphaned_hvg_encoder_also_blocks_new_smoke_run(rig, monkeypatch):
    path = rig.root / "results/pending_workflow/login-hvg.json"
    path.write_text(json.dumps({"root": str(rig.root), "current": {
        "process_group": 12345, "host": "login23-g-1.example"}}))
    monkeypatch.setattr(smoke.controls.os, "killpg", lambda *args: None, raising=False)
    with pytest.raises(RuntimeError, match="still active"):
        execute(rig)
    assert not rig.calls


def test_cache_setup_time_is_counted_against_diagnostic_deadline(rig, monkeypatch):
    clock = iter([100, 101, 110, 701])
    monkeypatch.setattr(smoke.time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="budget is exhausted"):
        execute(rig)
    assert len(rig.calls) == 1
