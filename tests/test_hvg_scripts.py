"""Exercise the all-model HVG profile with fake Slurm and preserved baselines."""
from __future__ import annotations

import os
import shutil

import pytest

from test_pending_scripts import invoke, pending_cluster, records, submitted

pytestmark = pytest.mark.skipif(os.name == "nt" or shutil.which("zsh") is None, reason="Requires POSIX and zsh")
STATE = "results/hvg_workflow"
PILOT = f"{STATE}/pilot/timings.json"


@pytest.fixture
def hvg_cluster(pending_cluster):
    state = pending_cluster[0] / STATE
    state.mkdir(parents=True)
    (state / "baseline-snapshot.json").write_text("{}\n")
    return pending_cluster


def test_hvg_setup_reuses_all_existing_environments_and_skips_encoders(hvg_cluster):
    result = invoke(hvg_cluster, "setup", script="hvg.sh", CPP_PYTHON="does-not-exist", CPP_PYTHON_UCE="does-not-exist")
    assert result.returncode == 0, result.stderr
    calls = records(hvg_cluster)
    assert [call["command"] for call in calls] == ["state", "run", "state"]
    assert calls[0]["args"] == ["snapshot", "--root", str(hvg_cluster[0]), "--config", "configs/custom config.yaml",
                                "--state", f"{STATE}/baseline-snapshot.json"]
    assert calls[1]["args"] == ["prefetch", "--models", "limix", "kumo"]
    assert "hvg.sh pilots" in result.stdout
    assert submitted(hvg_cluster) == []


def test_hvg_setup_bootstraps_only_missing_predictor_without_python311(hvg_cluster):
    (hvg_cluster[0] / ".venvs/limix/bin/python").unlink()
    result = invoke(hvg_cluster, "setup", script="hvg.sh", CPP_PYTHON_UCE="does-not-exist")
    assert result.returncode == 0, result.stderr
    calls = records(hvg_cluster)
    assert [call["args"] for call in calls if call["command"] == "bootstrap"] == [["limix"]]
    assert not any(call["command"] == "python311" for call in calls)


def test_hvg_pilots_use_both_inputs_and_serialize_all_three_models(hvg_cluster):
    result = invoke(hvg_cluster, "pilots", script="hvg.sh", MOCK_EMBEDDINGS_MISSING="1")
    assert result.returncode == 0, result.stderr
    calls = submitted(hvg_cluster)
    assert len(calls) == 3
    for index, model in enumerate(("tabpfn", "limix", "kumo")):
        args = calls[index]["args"]
        assert args[-7:] == ["cluster/pilot.sbatch", model, "--representations", "hvg", "hvg_pca128", "--output", PILOT]
        assert "--export=ALL" in args
        if index:
            assert f"--dependency=afterok:{100 + index}" in args
        else:
            assert not any(arg.startswith("--dependency") for arg in args)
    assert (hvg_cluster[0] / STATE / "slurm-jobs.tsv").read_text() == "pilot_tabpfn\t101\npilot_limix\t102\npilot_kumo\t103\n"
    assert not any("--embedding" in call["args"] for call in records(hvg_cluster))


def test_hvg_preview_has_120_tasks_and_fixed_concurrency(hvg_cluster):
    result = invoke(hvg_cluster, "submit", script="hvg.sh", CPP_ARRAY_CONCURRENCY="20")
    assert result.returncode == 0, result.stderr
    assert "0-119%4" in result.stdout
    assert "Jobs: 120; target/fold fits: 2040" in result.stdout
    assert [call["args"] for call in records(hvg_cluster) if call["command"] == "run"] == [
        ["jobs", "--stage", "hvg", "--output", "jobs/hvg.tsv"],
    ]
    assert submitted(hvg_cluster) == []


def test_hvg_execute_checks_own_pilot_without_requiring_embeddings(hvg_cluster):
    result = invoke(hvg_cluster, "submit", "--execute", script="hvg.sh", MOCK_EMBEDDINGS_MISSING="1")
    assert result.returncode == 0, result.stderr
    [call] = submitted(hvg_cluster)
    assert "--array=0-119%4" in call["args"]
    assert call["args"][-2:] == ["cluster/evaluate.sbatch", "jobs/hvg.tsv"]
    [budget] = [call for call in records(hvg_cluster) if call["command"] == "run" and call["args"][0] == "budget"]
    assert budget["args"] == ["budget", "--jobs", "jobs/hvg.tsv", "--pilot", PILOT, "--require-within-budget"]
    assert (hvg_cluster[0] / STATE / "slurm-jobs.tsv").read_text() == "production\t101\n"


def test_hvg_execute_refuses_insufficient_quota(hvg_cluster):
    result = invoke(hvg_cluster, "submit", "--execute", script="hvg.sh", MOCK_BUDGET_FAIL="1")
    assert result.returncode != 0
    assert "current remaining quota" in result.stdout
    assert submitted(hvg_cluster) == []


@pytest.mark.parametrize("count, succeeds", [(120, True), (119, False), (0, False)])
def test_hvg_delayed_report_checks_exactly_its_120_tasks(hvg_cluster, count, succeeds):
    (hvg_cluster[0] / STATE / "slurm-jobs.tsv").write_text("production\t1001\n")
    result = invoke(hvg_cluster, "report", script="hvg.sh", MOCK_ACCOUNTING_SUCCESS="1", MOCK_ACCOUNTING_TASKS=str(count))
    assert (result.returncode == 0) == succeeds, result.stderr
    assert len(submitted(hvg_cluster)) == int(succeeds)
    if succeeds:
        assert not any(arg.startswith("--dependency") for arg in submitted(hvg_cluster)[0]["args"])
    else:
        assert "does not confirm all 120 tasks" in result.stderr


@pytest.mark.parametrize("action", ["setup", "pilots", "submit"])
def test_active_pending_workflow_blocks_hvg_mutations(hvg_cluster, action):
    (hvg_cluster[0] / "results/pending_workflow/slurm-jobs.tsv").write_text("production\t999\n")
    result = invoke(hvg_cluster, action, script="hvg.sh", MOCK_ACTIVE_IDS="999")
    assert result.returncode == 2
    assert "still queued/running" in result.stderr
    assert not any(call["command"] in {"run", "bootstrap", "sbatch"} for call in records(hvg_cluster))


def test_active_hvg_workflow_blocks_pending_setup(hvg_cluster):
    (hvg_cluster[0] / STATE / "slurm-jobs.tsv").write_text("pilot_tabpfn\t999\n")
    result = invoke(hvg_cluster, "setup", MOCK_ACTIVE_IDS="999")
    assert result.returncode == 2
    assert "still queued/running" in result.stderr
    assert not any(call["command"] in {"run", "bootstrap", "sbatch"} for call in records(hvg_cluster))


def test_hvg_rejects_embedding_stage_before_mutations(hvg_cluster):
    result = invoke(hvg_cluster, "embeddings", script="hvg.sh")
    assert result.returncode == 2
    assert "no embedding jobs are needed" in result.stderr
    assert records(hvg_cluster) == []
