"""Run real shell wrappers against fake Slurm commands; no data, models or jobs."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ZSH = shutil.which("zsh")
pytestmark = pytest.mark.skipif(os.name == "nt" or ZSH is None, reason="Requires POSIX and zsh")


@pytest.fixture
def cluster(tmp_path):
    root = tmp_path / "project with spaces"
    shutil.copytree(Path(__file__).resolve().parents[1] / "cluster", root / "cluster")
    (root / "configs").mkdir()
    (root / "configs/custom config.yaml").write_text("{}\n")
    (root / "jobs").mkdir()
    (root / "jobs/test.tsv").write_text("predictor\ntabpfn\n")
    for name in ("core", "tabpfn"):
        interpreter = root / f".venvs/{name}/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)

    log = root / "calls.jsonl"
    helper = root / "record.py"
    helper.write_text(
        "import json, os, pathlib, sys\n"
        f"log = pathlib.Path({str(log)!r})\n"
        "rows = log.read_text().splitlines() if log.exists() else []\n"
        "record = {'command': sys.argv[1], 'args': sys.argv[2:],\n"
        "          'env': {k: os.environ.get(k) for k in\n"
        "                  ('CPP_ROOT', 'CPP_CONFIG', 'CPP_ENV', 'SLURM_EXPORT_ENV')}}\n"
        "with log.open('a') as stream: stream.write(json.dumps(record) + '\\n')\n"
        "if sys.argv[1] == 'sbatch': print(f'{101 + len(rows)};testcluster')\n"
    )
    mock_bin = root / "mock-bin"
    mock_bin.mkdir()

    def executable(path, source):
        path.write_text("#!/bin/sh\nset -eu\n" + source)
        path.chmod(0o755)

    runner = f"exec {shlex.quote(sys.executable)} {shlex.quote(str(helper))}"
    executable(mock_bin / "sbatch", runner + ' sbatch "$@"\n')
    executable(mock_bin / "srun", '[ "$1" = "--export=ALL" ]\nshift\nexec "$@"\n')
    executable(root / "cluster/run.sh", runner + ' run "$@"\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CPP_", "SLURM_"))}
    env.update(
        PATH=f"{mock_bin}{os.pathsep}{env.get('PATH', '')}",
        SLURM_SUBMIT_DIR=str(root),
        SLURM_ARRAY_TASK_ID="0",
        SBATCH_EXPORT="NONE",
        SLURM_EXPORT_ENV="NONE",
        SRUN_EXPORT_ENV="NONE",
    )
    return root, env, log


def invoke(cluster, script, *args, **updates):
    root, env, _ = cluster
    return subprocess.run(
        [ZSH, str(root / "cluster" / script), *args],
        cwd=root, env={**env, **updates}, text=True, capture_output=True, timeout=20,
    )


def records(cluster):
    log = cluster[2]
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


@pytest.mark.parametrize("custom_config", [False, True])
@pytest.mark.parametrize(
    "script,args,expected_args,environment",
    [
        ("stage.sbatch", ["prepare"], ["prepare"], None),
        ("pilot.sbatch", ["tabpfn", "--targets", "1"],
         ["pilot", "--predictors", "tabpfn", "--targets", "1"], "tabpfn"),
        ("encoder.sbatch", ["uce"], ["embed", "--encoder", "uce", "--device", "cuda"], "uce"),
        ("evaluate.sbatch", ["jobs/test.tsv"],
         ["array-task", "--jobs", "jobs/test.tsv", "--device", "cuda"], "tabpfn"),
    ],
)
def test_batch_wrappers_recover_missing_root_and_export_step_environment(
    cluster, script, args, expected_args, environment, custom_config,
):
    updates = {"CPP_CONFIG": "configs/custom config.yaml"} if custom_config else {}
    result = invoke(cluster, script, *args, **updates)
    assert result.returncode == 0, result.stderr
    [call] = records(cluster)
    assert call["command"] == "run"
    assert call["args"] == expected_args
    assert call["env"] == {
        "CPP_ROOT": str(cluster[0]),
        "CPP_CONFIG": updates.get("CPP_CONFIG", "configs/benchmark.yaml"),
        "CPP_ENV": environment,
        "SLURM_EXPORT_ENV": "ALL",
    }


def test_explicit_root_takes_precedence_over_submission_directory(cluster):
    root, env, _ = cluster
    env["SLURM_SUBMIT_DIR"] = str(root / "elsewhere")
    result = invoke(cluster, "stage.sbatch", "prepare", CPP_ROOT=str(root))
    assert result.returncode == 0, result.stderr
    assert records(cluster)[0]["env"]["CPP_ROOT"] == str(root)


def test_wrong_submission_directory_fails_before_srun(cluster):
    root, env, _ = cluster
    env["SLURM_SUBMIT_DIR"] = str(root / "elsewhere")
    result = invoke(cluster, "stage.sbatch", "prepare")
    assert result.returncode == 2
    assert "Project directory not found" in result.stderr
    assert records(cluster) == []


@pytest.mark.parametrize("custom_config", [False, True])
def test_pilot_launcher_exports_environment_and_preserves_dependency_chain(cluster, custom_config):
    updates = {"CPP_ROOT": str(cluster[0])}
    if custom_config:
        updates["CPP_CONFIG"] = "configs/custom config.yaml"
    result = invoke(cluster, "start.sh", "pilot", **updates)
    assert result.returncode == 0, result.stderr
    calls = records(cluster)
    assert len(calls) == 3
    for call in calls:
        assert call["command"] == "sbatch"
        assert "--export=ALL" in call["args"]
        assert call["env"]["CPP_ROOT"] == str(cluster[0])
        assert call["env"]["CPP_CONFIG"] == updates.get("CPP_CONFIG", "configs/benchmark.yaml")
        assert call["env"]["CPP_ENV"] == "core"
    assert "--dependency=afterok:101" in calls[1]["args"]
    assert "--dependency=afterok:102" in calls[2]["args"]


def test_cpu_array_submission_is_dry_by_default_and_exports_config(cluster):
    root = cluster[0]
    (root / "jobs/test.tsv").write_text("predictor\nmean\n")
    updates = {"CPP_ROOT": str(root), "CPP_CONFIG": "configs/custom config.yaml"}
    preview = invoke(cluster, "submit.sh", "jobs/test.tsv", "--cpu", **updates)
    assert preview.returncode == 0, preview.stderr
    assert "--export=ALL" in preview.stdout
    assert records(cluster) == []
    result = invoke(cluster, "submit.sh", "jobs/test.tsv", "--cpu", "--execute", **updates)
    assert result.returncode == 0, result.stderr
    [call] = records(cluster)
    assert call["command"] == "sbatch"
    assert {"--export=ALL", "--array=0-0%4", "--partition=c23mm", "--gres=none"} <= set(call["args"])
    assert call["env"]["CPP_ROOT"] == str(root)
    assert call["env"]["CPP_CONFIG"] == updates["CPP_CONFIG"]
