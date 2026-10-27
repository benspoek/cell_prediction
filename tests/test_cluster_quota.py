"""Quota observations gate launches without changing scientific configuration."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest
import yaml

from cell_property_prediction.config import compute_config, load_config
from cell_property_prediction.jobs import compute_fingerprint

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cluster_quota", ROOT / "cluster/quota.py")
quota = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quota)

# Actual r_wlm_usage report: CPU remainder, percentage, and total are decoys.
REPORT = """Account:                                        thes2301
Type:                                               thes
Start of Accounting Period:                   03.06.2026
End of Accounting Period:                     07.11.2026
State of project:                                 active
--------------------------------------------------------
Partitiontype:                                       HPC
 Quota last month (core-h):                         9305
 Quota current month (core-h):                      9305
 Quota next month (core-h):                         1861
 Remaining core-h of prev. month:                   9269
 Consumed core-h current month:                        0
 Consumed core-h last 4 weeks:                         0
 Consumable core-h (%):                              199
 Consumable core-h:                                20436
 Total quota (core-h):                             47999
 Total consumed core-h so far:                        72
--------------------------------------------------------
                                                    ^
--------------------------------------------------------
Partitiontype:                                        ML
 Quota last month (gpu-h):                           291
 Quota current month (gpu-h):                        291
 Quota next month (gpu-h):                            58
 Remaining gpu-h of prev. month:                     291
 Consumed gpu-h current month:                         0
 Consumed gpu-h last 4 weeks:                          0
 Consumable gpu-h (%):                              200
 Consumable gpu-h:                                  640
 Total quota (gpu-h):                              1500
 Total consumed gpu-h so far:                          0
--------------------------------------------------------
Allowed partitions:                     c23g,c23ml,c23mm,c23ms
Max. allowed wallclocktime (hours):                 24.0
Max. allowed cores per job:                           96
"""
ABSOLUTE_ROW = " Consumable gpu-h:                                  640"
ML_HEADER = "Partitiontype:                                        ML"


@pytest.mark.parametrize("report", [
    REPORT,
    REPORT.replace("\n", "\r\n"),
    "\x1b[32m" + REPORT + "\x1b[0m",
    REPORT.replace("Partitiontype:                                       HPC", "=== HPC partitions (CPU Core-h) ===")
          .replace(ML_HEADER, "=== ML partitions (GPU-h) ==="),
])
def test_parser_reads_only_absolute_consumable_ml_gpu_hours(report):
    assert quota.parse_quota(report, "thes2301") == 640


@pytest.mark.parametrize("value,expected", [("0", 0), ("63.25", 63.25), ("1,234.50", 1234.5)])
def test_parser_preserves_zero_decimal_and_grouped_hours(value, expected):
    assert quota.parse_quota(REPORT.replace(ABSOLUTE_ROW, f" Consumable gpu-h: {value}"), "thes2301") == expected


@pytest.mark.parametrize("report", [
    REPORT.replace("thes2301", "other-account"),
    REPORT.replace("Account:", "Project:"),
    REPORT + "Account: thes2301\n",
    REPORT.replace(ML_HEADER, "Partitiontype: HPC"),
    REPORT.replace(ML_HEADER, "=== ML partitions (CPU Core-h) ==="),
    REPORT.replace(ABSOLUTE_ROW, ""),
    REPORT.replace(ABSOLUTE_ROW, " Consumable hours: 640"),
    REPORT.replace(ABSOLUTE_ROW, " Consumable gpu-h (%): 640"),
    REPORT + "Partitiontype: ML\nConsumable gpu-h: 2\n",
    REPORT.replace(ABSOLUTE_ROW, ABSOLUTE_ROW + "\nConsumable gpu-h: 640"),
])
def test_parser_rejects_missing_mismatched_or_ambiguous_allowances(report):
    with pytest.raises(quota.QuotaError):
        quota.parse_quota(report, "thes2301")


@pytest.mark.parametrize("value", ["-1", "nan", "NaN", "inf", "Infinity", "1e3", "640 %", "640 hours", "12,34", ""])
def test_parser_rejects_invalid_absolute_numbers(value):
    with pytest.raises(quota.QuotaError):
        quota.parse_quota(REPORT.replace(ABSOLUTE_ROW, f"Consumable gpu-h: {value}"), "thes2301")


def test_query_uses_account_argument_timeout_and_c_locale(monkeypatch):
    calls = []
    monkeypatch.setenv("QUOTA_TEST_SENTINEL", "preserved")

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, stdout=REPORT, stderr="")

    monkeypatch.setattr(quota.subprocess, "run", run)
    result = quota.query_quota("thes2301")
    assert result["remaining_gpu_hours"] == 640
    assert result["source"] == "r_wlm_usage"
    assert result["account"] == "thes2301"
    assert result["output_sha256"] == hashlib.sha256(REPORT.encode()).hexdigest()
    assert datetime.fromisoformat(result["checked_at"]).utcoffset().total_seconds() == 0
    [(arguments, options)] = calls
    assert arguments == ["r_wlm_usage", "-q", "-p", "thes2301"]
    assert options["capture_output"] and options["text"] and options["check"]
    assert 0 < options["timeout"] <= 30
    assert not options.get("shell", False)
    assert options["env"]["LC_ALL"] == options["env"]["LANG"] == "C"
    assert options["env"]["QUOTA_TEST_SENTINEL"] == "preserved"


@pytest.mark.parametrize("error", [
    FileNotFoundError("r_wlm_usage missing"),
    subprocess.CalledProcessError(1, ["r_wlm_usage"], stderr="service unavailable"),
    subprocess.TimeoutExpired(["r_wlm_usage"], 30),
])
def test_query_failure_does_not_supply_a_cached_or_configured_allowance(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(quota.subprocess, "run", fail)
    with pytest.raises(quota.QuotaError, match="No configured or cached allowance"):
        quota.query_quota("thes2301")


@pytest.mark.parametrize("account", [None, "", "wrong account", "-q", "thes2301;echo bad"])
def test_query_rejects_invalid_accounts_before_launching_a_command(monkeypatch, account):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid account must not launch a subprocess")
    monkeypatch.setattr(quota.subprocess, "run", forbidden)
    with pytest.raises(quota.QuotaError):
        quota.query_quota(account)


@pytest.fixture
def configured_project(tmp_path):
    root = tmp_path / "project with spaces"
    config = load_config(ROOT / "configs/benchmark.yaml")
    config["cluster"]["remaining_gpu_hours"] = 9999
    path = root / "configs/local config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("# Preserve user formatting and comments.\n" + yaml.safe_dump(config), encoding="utf-8")
    source = root / "src/cell_property_prediction/unchanged.py"
    source.parent.mkdir(parents=True)
    source.write_text("# scientific source stays identical\n", encoding="utf-8")
    return root, path, config


def observation(hours=640):
    return {"source": "r_wlm_usage", "account": "thes2301", "remaining_gpu_hours": hours,
            "checked_at": "2026-10-27T12:00:00+00:00"}


@pytest.mark.parametrize("command", ["budget", "hvg-encoder-budget"])
def test_budget_injects_only_live_allowance_preserves_fingerprints_and_removes_temp_config(
    configured_project, monkeypatch, capsys, command,
):
    root, path, original = configured_project
    before = path.read_bytes()
    signature = compute_fingerprint(original, root)
    seen_paths = []
    calls = []
    monkeypatch.setattr(quota, "query_quota", lambda account: observation())

    def invoke(arguments):
        calls.append(arguments)
        runtime_path = Path(arguments[arguments.index("--config") + 1])
        seen_paths.append(runtime_path)
        assert runtime_path != path
        runtime = load_config(runtime_path)
        assert runtime["cluster"]["remaining_gpu_hours"] == 640
        assert original["cluster"]["remaining_gpu_hours"] == 9999
        assert compute_config(runtime) == compute_config(original)
        assert compute_fingerprint(runtime, root) == signature
        assert path.read_bytes() == before
        print(json.dumps({"within_budget": True, "remaining_gpu_hours": 640}))
        return 0

    monkeypatch.setattr(quota, "_invoke_cli", invoke)
    arguments = ["--require-within-budget"]
    assert quota.run_budget(root, path.relative_to(root), command, arguments) == 0
    [called] = calls
    assert called[:2] == ["--root", str(root)]
    assert called[4:] == [command, *arguments]
    assert not seen_paths[0].exists()
    assert path.read_bytes() == before
    result = json.loads(capsys.readouterr().out)
    assert result["within_budget"] is True
    assert result["quota"] == observation()


def test_each_budget_check_reads_a_fresh_live_observation(configured_project, monkeypatch, capsys):
    root, path, _ = configured_project
    observations = iter([observation(640), observation(500)])
    monkeypatch.setattr(quota, "query_quota", lambda account: next(observations))
    monkeypatch.setattr(quota, "_invoke_cli", lambda args: 0)
    assert quota.run_budget(root, path, "budget", []) == 0
    assert json.loads(capsys.readouterr().out)["quota"]["remaining_gpu_hours"] == 640
    assert quota.run_budget(root, path, "budget", []) == 0
    assert json.loads(capsys.readouterr().out)["quota"]["remaining_gpu_hours"] == 500


def test_manual_allowance_requires_explicit_source_and_never_queries_cluster(configured_project, monkeypatch, capsys):
    root, path, _ = configured_project
    def forbidden(account):
        pytest.fail("Explicit config source must not query the cluster")
    monkeypatch.setattr(quota, "query_quota", forbidden)
    monkeypatch.setattr(quota, "_invoke_cli", lambda args: 0)
    assert quota.run_budget(root, path, "budget", [], source="config") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["quota"]["source"] == "config"
    assert result["quota"]["remaining_gpu_hours"] == 9999


@pytest.mark.parametrize("value", [None, -1, float("nan"), float("inf"), True, "unknown"])
def test_explicit_manual_source_rejects_invalid_allowance(configured_project, value):
    root, path, config = configured_project
    config["cluster"]["remaining_gpu_hours"] = value
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(quota.QuotaError):
        quota.run_budget(root, path, "quota", [], source="config")


def test_live_query_failure_never_falls_back_to_nonempty_config(configured_project, monkeypatch):
    root, path, _ = configured_project
    def fail(account):
        raise quota.QuotaError("live quota service unavailable")
    def forbidden(arguments):
        pytest.fail("Budget CLI cannot run without an observed allowance")
    monkeypatch.setattr(quota, "query_quota", fail)
    monkeypatch.setattr(quota, "_invoke_cli", forbidden)
    with pytest.raises(quota.QuotaError, match="live quota service unavailable"):
        quota.run_budget(root, path, "budget", ["--require-within-budget"])


@pytest.mark.parametrize("failure", [RuntimeError("Set the actual remaining GPU hours before submission"), SystemExit(2)])
def test_failed_budget_remains_failed_includes_observation_and_cleans_temp_file(
    configured_project, monkeypatch, capsys, failure,
):
    root, path, _ = configured_project
    before = path.read_bytes()
    runtime_paths = []
    monkeypatch.setattr(quota, "query_quota", lambda account: observation())
    def invoke(arguments):
        runtime_paths.append(Path(arguments[arguments.index("--config") + 1]))
        print(json.dumps({"within_budget": False, "remaining_gpu_hours": 640}))
        raise failure
    monkeypatch.setattr(quota, "_invoke_cli", invoke)
    assert quota.run_budget(root, path, "budget", ["--require-within-budget"]) == 2
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["within_budget"] is False
    assert result["quota"] == observation()
    assert result["error"]
    assert captured.err
    assert not runtime_paths[0].exists()
    assert path.read_bytes() == before


def test_quota_command_reports_observation_without_invoking_budget_or_creating_temp_file(
    configured_project, monkeypatch, capsys,
):
    root, path, _ = configured_project
    monkeypatch.setattr(quota, "query_quota", lambda account: observation())
    def forbidden(*args, **kwargs):
        pytest.fail("Quota inspection must not invoke the budget or create a runtime config")
    monkeypatch.setattr(quota, "_invoke_cli", forbidden)
    monkeypatch.setattr(quota.tempfile, "TemporaryDirectory", forbidden)
    assert quota.run_budget(root, path, "quota", []) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {"remaining_gpu_hours": 640, "quota": observation()}


def test_main_reports_lookup_failure_and_explicit_manual_fallback(configured_project, monkeypatch, capsys):
    root, path, _ = configured_project
    monkeypatch.delenv("CPP_QUOTA_SOURCE", raising=False)
    def fail(account):
        raise quota.QuotaError("quota service unavailable")
    monkeypatch.setattr(quota, "query_quota", fail)
    assert quota.main(["--root", str(root), "--config", str(path), "quota"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "quota service unavailable" in captured.err
    assert "CPP_QUOTA_SOURCE=config" in captured.err


@pytest.mark.skipif(os.name == "nt" or shutil.which("zsh") is None, reason="Requires POSIX and zsh")
@pytest.mark.parametrize("command", ["quota", "budget", "hvg-encoder-budget", "preflight"])
def test_run_sh_routes_quota_gates_to_wrapper_and_other_commands_to_existing_cli(tmp_path, command):
    root = tmp_path / "project with spaces"
    (root / "cluster").mkdir(parents=True)
    shutil.copyfile(ROOT / "cluster/run.sh", root / "cluster/run.sh")
    log = root / "arguments.json"
    recorder = root / "record.py"
    recorder.write_text("import json, pathlib, sys\n" +
                        f"pathlib.Path({str(log)!r}).write_text(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
    interpreter = root / ".venvs/core/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " +
                           shlex.quote(str(recorder)) + ' "$@"\n', encoding="utf-8")
    interpreter.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("CPP_")}
    env.update(CPP_ROOT=str(root), CPP_CONFIG="configs/custom config.yaml")
    result = subprocess.run([shutil.which("zsh"), str(root / "cluster/run.sh"), command, "--flag", "value with spaces"],
                            cwd=root, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    args = json.loads(log.read_text())
    prefix = [str(root / "cluster/quota.py")] if command != "preflight" else ["-m", "cell_property_prediction.cli"]
    assert args == [*prefix, "--root", str(root), "--config", "configs/custom config.yaml",
                    command, "--flag", "value with spaces"]
