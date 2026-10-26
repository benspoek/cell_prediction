"""Exercise continuation launchers with fake Slurm; no models, data, or real jobs."""
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
STATE = "results/pending_workflow"
PILOT = f"{STATE}/pilot/timings.json"


@pytest.fixture
def pending_cluster(tmp_path):
    root = tmp_path / "existing project with spaces"
    shutil.copytree(Path(__file__).resolve().parents[1] / "cluster", root / "cluster")
    (root / "configs").mkdir()
    (root / "configs/custom config.yaml").write_text("{}\n")
    (root / STATE).mkdir(parents=True)
    (root / STATE / "baseline-snapshot.json").write_text("{}\n")
    for relative in ("data/raw/original", "data/processed/original", "data/splits/original",
                     "results/cell_kfold/pca128/tabpfn/original"):
        marker = root / relative
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("preserve this baseline\n")
    for name in ("core", "tabpfn", "limix", "kumo", "uce", "nicheformer"):
        interpreter = root / f".venvs/{name}/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)

    log = root / "calls.jsonl"
    helper = root / "record.py"
    helper.write_text(
        "import json, os, pathlib, sys\n"
        f"root = pathlib.Path({str(root)!r})\n"
        f"log = pathlib.Path({str(log)!r})\n"
        "command, args = sys.argv[1], sys.argv[2:]\n"
        "record = {'command': command, 'args': args, 'env': {k: os.environ.get(k) for k in\n"
        "          ('CPP_ROOT', 'CPP_CONFIG', 'CPP_ENV', 'CPP_ARRAY_CONCURRENCY')}}\n"
        "with log.open('a') as stream: stream.write(json.dumps(record) + '\\n')\n"
        "if command == 'sbatch':\n"
        "    counter = root / 'slurm-counter'\n"
        "    value = int(counter.read_text()) + 1 if counter.exists() else 101\n"
        "    counter.write_text(str(value))\n"
        "    print(f'{value};testcluster' if '--parsable' in args else f'Submitted batch job {value}')\n"
        "elif command == 'squeue':\n"
        "    if os.environ.get('MOCK_QUEUE_FAIL'): sys.exit('controller unavailable')\n"
        "    state_file = root / 'results/pending_workflow/slurm-jobs.tsv'\n"
        "    saved = [line.split('\\t')[1] for line in state_file.read_text().splitlines()] if state_file.exists() else []\n"
        "    default = ','.join(dict.fromkeys(saved)) if os.environ.get('MOCK_QUEUE_STATES') else ''\n"
        "    active = os.environ.get('MOCK_ACTIVE_IDS', default).split(',')\n"
        "    for job in active:\n"
        "        if job: print(f'{job}|RUNNING')\n"
        "elif command == 'sacct' and os.environ.get('MOCK_ACCOUNTING_SUCCESS'):\n"
        "    job = args[args.index('-j') + 1]\n"
        "    count = int(os.environ.get('MOCK_ACCOUNTING_TASKS', '300'))\n"
        "    for index in range(count):\n"
        "        code = '1:0' if index == 2 and os.environ.get('MOCK_ACCOUNTING_BAD_EXIT') else '0:0'\n"
        "        print(f'{job}_{index}|COMPLETED|{code}')\n"
        "elif command == 'state':\n"
        "    if args[0] == 'snapshot':\n"
        "        if os.environ.get('MOCK_SNAPSHOT_FAIL'): sys.exit('baseline incomplete')\n"
        "        (root / args[args.index('--state') + 1]).write_text('{}\\n')\n"
        "    elif os.environ.get('MOCK_STATE_FAIL'): sys.exit('baseline modified')\n"
        "    missing = os.environ.get('MOCK_EMBEDDINGS_MISSING')\n"
        "    if '--embedding' in args:\n"
        "        encoder = args[args.index('--embedding') + 1]\n"
        "        missing = missing or encoder in os.environ.get('MOCK_MISSING_ENCODERS', '').split(',')\n"
        "    if ('--embeddings' in args or '--embedding' in args) and missing:\n"
        "        sys.exit('valid embeddings required')\n"
        "elif command == 'run':\n"
        "    if args[0] == 'jobs':\n"
        "        target = root / args[args.index('--output') + 1]\n"
        "        target.parent.mkdir(exist_ok=True)\n"
        "        target.write_text('predictor\\n' + 'limix\\n' * 300)\n"
        "    if args[0] == 'budget' and os.environ.get('MOCK_BUDGET_FAIL'):\n"
        "        print('measured estimate exceeds current remaining quota')\n"
        "        sys.exit(1)\n"
    )
    mock_bin = root / "mock-bin"
    mock_bin.mkdir()

    def executable(path, source):
        path.write_text("#!/bin/sh\nset -eu\n" + source)
        path.chmod(0o755)

    runner = f"exec {shlex.quote(sys.executable)} {shlex.quote(str(helper))}"
    for command in ("sbatch", "squeue", "sacct", "python312", "python311"):
        executable(mock_bin / command, runner + f' {command} "$@"\n')
    executable(root / "cluster/run.sh", runner + ' run "$@"\n')
    executable(root / "cluster/bootstrap.sh", runner + ' bootstrap "$@"\n')
    (root / "cluster/pending_state.py").write_text(
        "import runpy, sys\n"
        f"sys.argv = [{str(helper)!r}, 'state', *sys.argv[1:]]\n"
        f"runpy.run_path({str(helper)!r}, run_name='__main__')\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CPP_", "SLURM_", "MOCK_"))}
    env.update(
        PATH=f"{mock_bin}{os.pathsep}{env.get('PATH', '')}",
        CPP_ROOT=str(root),
        CPP_CONFIG="configs/custom config.yaml",
        CPP_PYTHON=str(mock_bin / "python312"),
        CPP_PYTHON_UCE=str(mock_bin / "python311"),
    )
    return root, env, log


def invoke(pending_cluster, *args, script="pending.sh", **updates):
    root, env, _ = pending_cluster
    return subprocess.run(
        [ZSH, str(root / "cluster" / script), *args], cwd=root,
        env={**env, **updates}, text=True, capture_output=True, timeout=20,
    )


def records(pending_cluster):
    log = pending_cluster[2]
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def save_ids(pending_cluster, *rows):
    path = pending_cluster[0] / STATE / "slurm-jobs.tsv"
    path.write_text("".join(f"{stage}\t{job}\n" for stage, job in rows))


def submitted(pending_cluster):
    return [call for call in records(pending_cluster) if call["command"] == "sbatch"]


def test_setup_checks_both_interpreters_and_snapshots_before_fetch(pending_cluster):
    root = pending_cluster[0]
    snapshot = root / STATE / "baseline-snapshot.json"
    snapshot.unlink()
    result = invoke(pending_cluster, "setup")
    assert result.returncode == 0, result.stderr
    calls = records(pending_cluster)
    assert [call["command"] for call in calls] == [
        "python312", "python311", "state", "run", "bootstrap", "bootstrap", "bootstrap", "bootstrap", "state",
    ]
    assert calls[0]["args"][-1] == "3.12"
    assert calls[1]["args"][-1] == "3.11"
    assert calls[2]["args"] == [
        "snapshot", "--root", str(root), "--config", "configs/custom config.yaml",
        "--state", f"{STATE}/baseline-snapshot.json",
    ]
    assert calls[3]["args"] == ["prefetch", "--models", "limix", "kumo", "genept", "uce", "nicheformer"]
    assert [call["args"] for call in calls if call["command"] == "bootstrap"] == [
        ["limix"], ["kumo"], ["uce"], ["nicheformer"],
    ]
    assert submitted(pending_cluster) == []
    for relative in ("data/raw/original", "data/processed/original", "data/splits/original",
                     "results/cell_kfold/pca128/tabpfn/original"):
        assert (root / relative).read_text() == "preserve this baseline\n"


def test_setup_missing_uce_python_stops_before_downloads_and_snapshot(pending_cluster):
    result = invoke(pending_cluster, "setup", CPP_PYTHON_UCE="/missing/python3.11")
    assert result.returncode == 2
    assert "Python 3.11" in result.stderr
    assert [call["command"] for call in records(pending_cluster)] == ["python312"]


def test_setup_requires_existing_tabpfn_environment(pending_cluster):
    (pending_cluster[0] / ".venvs/tabpfn/bin/python").unlink()
    result = invoke(pending_cluster, "setup")
    assert result.returncode == 2
    assert "existing core and tabpfn" in result.stderr
    assert records(pending_cluster) == []


def test_snapshot_failure_stops_before_prefetch_or_bootstrap(pending_cluster):
    result = invoke(pending_cluster, "setup", MOCK_SNAPSHOT_FAIL="1")
    assert result.returncode != 0
    assert "baseline incomplete" in result.stderr
    assert not any(call["command"] in {"run", "bootstrap", "sbatch"} for call in records(pending_cluster))


def test_embeddings_and_serial_pilots_use_all_three_success_dependencies(pending_cluster):
    result = invoke(pending_cluster, "embeddings")
    assert result.returncode == 0, result.stderr
    # Encoder jobs may still be running when pilots are submitted with dependencies.
    result = invoke(pending_cluster, "pilots", MOCK_QUEUE_STATES="RUNNING\n", MOCK_EMBEDDINGS_MISSING="1")
    assert result.returncode == 0, result.stderr
    calls = submitted(pending_cluster)
    assert len(calls) == 6
    assert calls[0]["args"][-6:] == ["cluster/stage.sbatch", "embed", "--encoder", "genept", "--device", "cpu"]
    assert calls[1]["args"][-2:] == ["cluster/encoder.sbatch", "uce"]
    assert calls[2]["args"][-2:] == ["cluster/encoder.sbatch", "nicheformer"]
    assert "--dependency=afterok:101:102:103" in calls[3]["args"]
    assert "--dependency=afterok:104" in calls[4]["args"]
    assert "--dependency=afterok:105" in calls[5]["args"]
    assert calls[3]["args"][-8:] == [
        "cluster/pilot.sbatch", "tabpfn", "--representations", "genept", "uce", "nicheformer", "--output", PILOT,
    ]
    for call, model in zip(calls[4:], ("limix", "kumo")):
        assert call["args"][-11:] == [
            "cluster/pilot.sbatch", model, "--representations", "pca128", "ion_channels", "authors25",
            "genept", "uce", "nicheformer", "--output", PILOT,
        ]
    for call in calls:
        assert {"--export=ALL", "--parsable"} <= set(call["args"])
        assert call["env"]["CPP_ROOT"] == str(pending_cluster[0])
        assert call["env"]["CPP_CONFIG"] == "configs/custom config.yaml"
        assert call["env"]["CPP_ENV"] == "core"
    assert not any("prepare" in call["args"] or "splits" in call["args"] for call in records(pending_cluster))


def test_pilots_without_encoder_ids_require_real_embedding_validation(pending_cluster):
    result = invoke(pending_cluster, "pilots", MOCK_EMBEDDINGS_MISSING="1")
    assert result.returncode != 0
    assert "valid embeddings required" in result.stderr
    assert submitted(pending_cluster) == []
    assert any("--embedding" in call["args"] for call in records(pending_cluster))


def test_pilots_can_reuse_valid_embeddings_without_saved_ids(pending_cluster):
    result = invoke(pending_cluster, "pilots")
    assert result.returncode == 0, result.stderr
    calls = submitted(pending_cluster)
    assert len(calls) == 3
    assert not any(arg.startswith("--dependency") for arg in calls[0]["args"])
    assert "--dependency=afterok:101" in calls[1]["args"]
    assert "--dependency=afterok:102" in calls[2]["args"]


def test_pilots_reject_missing_embedding_with_inactive_saved_job(pending_cluster):
    save_ids(pending_cluster, ("genept", "101"), ("uce", "102"))
    result = invoke(pending_cluster, "pilots", MOCK_EMBEDDINGS_MISSING="1")
    assert result.returncode == 2
    assert "no longer active and its embedding is invalid" in result.stderr
    assert submitted(pending_cluster) == []


def test_pilots_wait_only_for_unfinished_encoders_when_genept_is_complete(pending_cluster):
    save_ids(pending_cluster, ("genept", "101"), ("uce", "102"), ("nicheformer", "103"))
    result = invoke(
        pending_cluster, "pilots", MOCK_MISSING_ENCODERS="uce,nicheformer",
        MOCK_ACTIVE_IDS="102,103", MOCK_QUEUE_STATES="RUNNING\n",
    )
    assert result.returncode == 0, result.stderr
    assert "--dependency=afterok:102:103" in submitted(pending_cluster)[0]["args"]


def test_pilots_reuse_completed_embeddings_with_expired_saved_ids(pending_cluster):
    save_ids(pending_cluster, ("genept", "101"), ("uce", "102"), ("nicheformer", "103"))
    result = invoke(pending_cluster, "pilots")
    assert result.returncode == 0, result.stderr
    assert not any(arg.startswith("--dependency") for arg in submitted(pending_cluster)[0]["args"])
    assert not any(call["command"] == "squeue" for call in records(pending_cluster))


def test_submission_defaults_to_300_job_preview_without_budget_or_sbatch(pending_cluster):
    result = invoke(pending_cluster, "submit", CPP_ARRAY_CONCURRENCY="20")
    assert result.returncode == 0, result.stderr
    assert "0-299%4" in result.stdout
    assert "Jobs: 300; target/fold fits: 5100" in result.stdout
    calls = records(pending_cluster)
    assert [call["args"] for call in calls if call["command"] == "run"] == [
        ["jobs", "--stage", "pending", "--output", "jobs/pending.tsv"],
    ]
    assert submitted(pending_cluster) == []


def test_execute_uses_new_pilot_and_refuses_insufficient_current_budget(pending_cluster):
    result = invoke(pending_cluster, "submit", "--execute", MOCK_BUDGET_FAIL="1")
    assert result.returncode != 0
    assert "current remaining quota" in result.stdout
    budgets = [call for call in records(pending_cluster) if call["command"] == "run" and call["args"][0] == "budget"]
    assert [call["args"] for call in budgets] == [
        ["budget", "--jobs", "jobs/pending.tsv", "--pilot", PILOT, "--require-within-budget"],
    ]
    assert submitted(pending_cluster) == []


def test_execute_requires_valid_embeddings_before_budget_and_sbatch(pending_cluster):
    result = invoke(pending_cluster, "submit", "--execute", MOCK_EMBEDDINGS_MISSING="1")
    assert result.returncode != 0
    assert "valid embeddings required" in result.stderr
    assert not any(call["command"] == "run" and call["args"][0] == "budget" for call in records(pending_cluster))
    assert submitted(pending_cluster) == []


def test_execute_saves_array_id_and_report_waits_for_its_success(pending_cluster):
    result = invoke(pending_cluster, "submit", "--execute")
    assert result.returncode == 0, result.stderr
    assert "Submitted batch job 101" in result.stdout
    state = pending_cluster[0] / STATE / "slurm-jobs.tsv"
    assert state.read_text() == "production\t101\n"
    result = invoke(pending_cluster, "report", MOCK_QUEUE_STATES="RUNNING\n")
    assert result.returncode == 0, result.stderr
    calls = submitted(pending_cluster)
    assert len(calls) == 2
    assert "--array=0-299%4" in calls[0]["args"]
    assert "--export=ALL" in calls[0]["args"]
    assert calls[0]["args"][-2:] == ["cluster/evaluate.sbatch", "jobs/pending.tsv"]
    assert "--dependency=afterok:101" in calls[1]["args"]
    assert calls[1]["args"][-2:] == ["cluster/stage.sbatch", "report"]
    assert state.read_text() == "production\t101\nreport\t102\n"


def test_delayed_report_verifies_every_array_task_without_expired_dependency(pending_cluster):
    save_ids(pending_cluster, ("production", "1001"))
    result = invoke(pending_cluster, "report", MOCK_ACCOUNTING_SUCCESS="1")
    assert result.returncode == 0, result.stderr
    [call] = submitted(pending_cluster)
    assert not any(arg.startswith("--dependency") for arg in call["args"])
    [accounting] = [call for call in records(pending_cluster) if call["command"] == "sacct"]
    assert accounting["args"] == [
        "-X", "--array", "-n", "-P", "-j", "1001", "--format=JobID%100,State%30,ExitCode",
    ]


@pytest.mark.parametrize("successful_tasks", ["0", "299"])
def test_delayed_report_refuses_incomplete_or_missing_accounting(pending_cluster, successful_tasks):
    save_ids(pending_cluster, ("production", "1001"))
    result = invoke(
        pending_cluster, "report", MOCK_ACCOUNTING_SUCCESS="1", MOCK_ACCOUNTING_TASKS=successful_tasks,
    )
    assert result.returncode != 0
    assert "does not confirm all 300 tasks" in result.stderr
    assert submitted(pending_cluster) == []


def test_delayed_report_refuses_completed_task_with_nonzero_exit(pending_cluster):
    save_ids(pending_cluster, ("production", "1001"))
    result = invoke(pending_cluster, "report", MOCK_ACCOUNTING_SUCCESS="1", MOCK_ACCOUNTING_BAD_EXIT="1")
    assert result.returncode != 0
    assert "does not confirm all 300 tasks" in result.stderr
    assert submitted(pending_cluster) == []


def test_report_without_saved_array_id_fails_before_submission(pending_cluster):
    result = invoke(pending_cluster, "report")
    assert result.returncode == 2
    assert "No saved production array ID" in result.stderr
    assert submitted(pending_cluster) == []


@pytest.mark.parametrize("stage", ["setup", "embeddings", "pilots", "submit"])
def test_active_workflow_cannot_be_rebuilt_or_resubmitted(pending_cluster, stage):
    save_ids(pending_cluster, ("production", "101"))
    manifest = pending_cluster[0] / "jobs/pending.tsv"
    manifest.parent.mkdir()
    manifest.write_text("do not rewrite active array input\n")
    result = invoke(pending_cluster, stage, MOCK_QUEUE_STATES="RUNNING\n")
    assert result.returncode == 2
    assert "still queued/running" in result.stderr
    assert not any(call["command"] in {"run", "bootstrap", "sbatch"} for call in records(pending_cluster))
    assert manifest.read_text() == "do not rewrite active array input\n"


def test_old_active_job_is_checked_even_when_a_newer_id_was_recorded(pending_cluster):
    save_ids(pending_cluster, ("genept", "101"), ("genept", "102"))
    result = invoke(pending_cluster, "embeddings", MOCK_QUEUE_STATES="PENDING\n")
    assert result.returncode == 2
    [query] = [call for call in records(pending_cluster) if call["command"] == "squeue"]
    assert query["args"] == ["-h", "--me", "-o", "%F|%T"]
    assert submitted(pending_cluster) == []


def test_unrelated_active_jobs_do_not_block_the_continuation(pending_cluster):
    save_ids(pending_cluster, ("production", "101"))
    result = invoke(pending_cluster, "submit", MOCK_ACTIVE_IDS="999", MOCK_QUEUE_STATES="RUNNING\n")
    assert result.returncode == 0, result.stderr
    assert "0-299%4" in result.stdout
    assert submitted(pending_cluster) == []


def test_queue_query_failure_blocks_mutations(pending_cluster):
    save_ids(pending_cluster, ("production", "101"))
    result = invoke(pending_cluster, "embeddings", MOCK_QUEUE_FAIL="1")
    assert result.returncode == 2
    assert "Could not check saved jobs" in result.stderr
    assert submitted(pending_cluster) == []


def test_changed_baseline_blocks_new_jobs(pending_cluster):
    result = invoke(pending_cluster, "embeddings", MOCK_STATE_FAIL="1")
    assert result.returncode != 0
    assert "baseline modified" in result.stderr
    assert submitted(pending_cluster) == []


def test_missing_snapshot_instructs_setup_before_work(pending_cluster):
    (pending_cluster[0] / STATE / "baseline-snapshot.json").unlink()
    result = invoke(pending_cluster, "embeddings")
    assert result.returncode == 2
    assert "pending.sh setup first" in result.stderr
    assert records(pending_cluster) == []


def test_status_queries_saved_ids_without_submission(pending_cluster):
    save_ids(pending_cluster, ("genept", "101"), ("production", "102"))
    result = invoke(pending_cluster, "status")
    assert result.returncode == 0, result.stderr
    calls = records(pending_cluster)
    assert [call["command"] for call in calls] == ["state", "squeue", "sacct"]
    assert calls[1]["args"] == ["-h", "--me", "-o", "%F|%T"]
    assert calls[2]["args"] == ["-j", "101,102", "--format=JobID,JobName,State,Elapsed,AllocTRES"]
    assert submitted(pending_cluster) == []


def test_submit_custom_pilot_path_with_spaces_passes_unchanged(pending_cluster):
    manifest = pending_cluster[0] / "jobs/test.tsv"
    manifest.parent.mkdir()
    manifest.write_text("predictor\ntabpfn\n")
    result = invoke(pending_cluster, "jobs/test.tsv", "--pilot", "results/custom pilot/timings.json", "--execute", script="submit.sh")
    assert result.returncode == 0, result.stderr
    [budget] = [call for call in records(pending_cluster) if call["command"] == "run"]
    assert budget["args"] == [
        "budget", "--jobs", "jobs/test.tsv", "--pilot", "results/custom pilot/timings.json", "--require-within-budget",
    ]


@pytest.mark.parametrize("args", [("--pilot",), ("--pilot", "--execute")])
def test_submit_rejects_missing_pilot_path(pending_cluster, args):
    result = invoke(pending_cluster, "jobs/test.tsv", *args, script="submit.sh")
    assert result.returncode == 2
    assert "--pilot requires" in result.stderr
    assert records(pending_cluster) == []
