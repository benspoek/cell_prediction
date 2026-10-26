"""Unified HVG encoder workflow: synthetic artifacts and mocked Slurm only."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from cell_property_prediction import cli
from cell_property_prediction.config import atomic_json, load_config
from cell_property_prediction.jobs import compute_fingerprint, create_jobs
from test_pending_scripts import invoke, pending_cluster, records, save_ids, submitted

POSIX = pytest.mark.skipif(os.name == "nt" or shutil.which("zsh") is None, reason="Requires POSIX and zsh")
ENCODERS = ("genept", "uce", "nicheformer")
HVG_INPUTS = tuple(f"hvg_{name}" for name in ENCODERS)


@pytest.fixture
def benchmark(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs/benchmark.yaml")
    for protocol in config["evaluation"]["protocols"]:
        atomic_json(tmp_path / config["paths"]["splits_dir"] / f"{protocol}.json", {"folds": [{}] * 10})
    return config


def test_encoder_stage_and_default_pilot_leave_base_config_frozen(tmp_path, benchmark):
    config_file = Path(__file__).parents[1] / "configs/benchmark.yaml"
    before = config_file.read_bytes()
    jobs = create_jobs(benchmark, tmp_path, "hvg_embeddings", tmp_path / "encoder.tsv")
    assert len(jobs) == 180
    assert {(j["representation"], j["predictor"]) for j in jobs} == {
        (representation, predictor) for representation in HVG_INPUTS for predictor in ("tabpfn", "limix", "kumo")
    }
    assert json.loads((tmp_path / "encoder.json").read_text())["target_fits"] == 3060
    assert len(cli.parser().parse_args(["pilot", "--predictors", "tabpfn"]).representations) == 11
    assert cli.parser().parse_args(["jobs", "--stage", "hvg_embeddings", "--output", "jobs.tsv"]).stage == "hvg_embeddings"
    assert config_file.read_bytes() == before


def test_pilot_fingerprint_hashes_declared_fold_artifacts_only(tmp_path, benchmark):
    directory = tmp_path / benchmark["paths"]["embeddings_dir"] / "hvg/cell_kfold/fold_00"
    directory.mkdir(parents=True)
    files = [directory / name for name in ("selection.json", "counts.h5ad", "genept.npz", "genept.audit.json")]
    previous = compute_fingerprint(benchmark, tmp_path)
    for path in files:
        path.write_bytes(b"initial")
        current = compute_fingerprint(benchmark, tmp_path)
        assert current != previous
        previous = current
    work = directory / "uce_work/cache.npz"
    work.parent.mkdir()
    work.write_bytes(b"mutable")
    (directory / "intermediate.npz").write_bytes(b"temporary")
    assert compute_fingerprint(benchmark, tmp_path) == previous
    work.write_bytes(b"changed")
    assert compute_fingerprint(benchmark, tmp_path) == previous
    files[-1].write_bytes(b"changed audit")
    assert compute_fingerprint(benchmark, tmp_path) != previous


def test_cli_native_embedding_dispatch_and_scope_validation(tmp_path, benchmark, monkeypatch):
    import yaml
    from cell_property_prediction import hvg_embeddings
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(benchmark))
    calls = []
    monkeypatch.setattr(hvg_embeddings, "generate_hvg_embedding", lambda *a, **kw: calls.append((a, kw)) or {"valid": True})
    base = ["--root", str(tmp_path), "--config", str(config_path), "embed", "--encoder", "genept"]
    assert cli.main([*base, "--gene-set", "hvg", "--protocol", "cell_kfold", "--fold", "0", "--device", "cpu"]) == 0
    assert calls[0][0] == ("genept", benchmark)
    assert calls[0][1] == {"root": tmp_path, "protocol": "cell_kfold", "fold": 0, "device": "cpu", "offline": True}
    for args in (["--gene-set", "hvg"], ["--protocol", "cell_kfold", "--fold", "0"]):
        with pytest.raises(ValueError, match="requires|require"):
            cli.main([*base, *args])


def test_cli_encoder_readiness_panels_and_budget_gate(tmp_path, benchmark, monkeypatch):
    import yaml
    from cell_property_prediction import hvg_embeddings
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(benchmark))
    base = ["--root", str(tmp_path), "--config", str(path)]
    calls = []
    monkeypatch.setattr(hvg_embeddings, "prepare_hvg_panels", lambda *a: calls.append("prepare") or {})
    monkeypatch.setattr(hvg_embeddings, "check_hvg_panels", lambda *a: calls.append("check") or {})
    monkeypatch.setattr(hvg_embeddings, "check_hvg_embeddings", lambda *a, **kw: calls.append(kw) or {})
    monkeypatch.setattr(hvg_embeddings, "encoding_budget", lambda *a: {"within_budget": False})
    assert cli.main([*base, "hvg-panels"]) == 0
    assert cli.main([*base, "hvg-panels", "--check"]) == 0
    assert cli.main([*base, "hvg-embedding-check", "--encoder", "uce", "--first-only"]) == 0
    assert calls == ["prepare", "check", {"encoder": "uce", "first_only": True}]
    assert cli.main([*base, "hvg-encoder-budget"]) == 0
    with pytest.raises(RuntimeError, match="remaining GPU hours"):
        cli.main([*base, "hvg-encoder-budget", "--require-within-budget"])


def test_cli_predictor_pilot_validates_all_embeddings_before_fingerprint(tmp_path, benchmark, monkeypatch):
    import yaml
    from cell_property_prediction import hvg_embeddings
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(benchmark))
    calls = []
    def check_all(*args, **kwargs):
        calls.append("all60")
        hvg_embeddings._HASH_CACHE[("pilot validation warmed cache",)] = "hash"
        return {}
    monkeypatch.setattr(hvg_embeddings, "check_hvg_embeddings", check_all)
    def evaluate(*args, **kwargs):
        assert calls == ["all60"]
        assert not hvg_embeddings._HASH_CACHE
        calls.append(kwargs)
        return {"targets": [{"status": "complete", "elapsed_seconds": 1} for _ in range(17)]}
    monkeypatch.setattr(cli, "evaluate_job", evaluate)
    assert cli.main(["--root", str(tmp_path), "--config", str(path), "pilot", "--predictors", "kumo",
                     "--representations", "hvg_uce", "--targets", "17"]) == 0
    assert calls[1]["protocol"] == "cell_kfold" and calls[1]["fold"] == 0


@POSIX
def test_panel_and_first_encoder_arrays_use_all_active_success_dependencies(pending_cluster):
    result = invoke(pending_cluster, "panels", MOCK_PANELS_MISSING="1")
    assert result.returncode == 0, result.stderr
    [panel] = submitted(pending_cluster)
    assert panel["args"][-2:] == ["cluster/stage.sbatch", "hvg-panels"]
    save_ids(pending_cluster, ("panels", "101"), ("genept", "201"), ("uce", "202"), ("nicheformer", "203"))
    result = invoke(pending_cluster, "encoder-pilot", MOCK_PANELS_MISSING="1", MOCK_EMBEDDINGS_MISSING="1",
                    MOCK_HVG_FIRST_MISSING="1", MOCK_ACTIVE_IDS="101,201,202,203")
    assert result.returncode == 0, result.stderr
    calls = submitted(pending_cluster)[1:]
    assert len(calls) == 3
    for call, encoder in zip(calls, ENCODERS):
        assert "--dependency=afterok:201:202:203:101" in call["args"]
        assert call["args"][-2:] == ["cluster/hvg_encoder_array.sbatch", encoder]
    assert "--array=0%4" in calls[0]["args"]
    assert {"--partition=c23mm", "--gres=none", "--cpus-per-task=8", "--mem=32G"} <= set(calls[0]["args"])
    assert all("--array=0%2" in call["args"] for call in calls[1:])


@POSIX
def test_full_embedding_jobs_are_reused_when_their_artifacts_validate(pending_cluster):
    save_ids(pending_cluster, ("genept", "101"), ("uce", "102"), ("nicheformer", "103"))
    result = invoke(pending_cluster, "embeddings")
    assert result.returncode == 0, result.stderr
    assert submitted(pending_cluster) == []
    assert len([call for call in records(pending_cluster) if "--embedding" in call["args"]]) == 3


@POSIX
def test_remaining_encoders_gate_first_and_submit_exact_missing_indices(pending_cluster):
    result = invoke(pending_cluster, "encoders", MOCK_REMAINING_INDICES="1,19")
    assert result.returncode == 0, result.stderr
    calls = submitted(pending_cluster)
    assert len(calls) == 3
    assert "--array=1,19%4" in calls[0]["args"]
    assert all("--array=1,19%2" in call["args"] for call in calls[1:])
    runs = [call["args"] for call in records(pending_cluster) if call["command"] == "run"]
    assert runs[:3] == [["hvg-panels", "--check"], ["hvg-embedding-check", "--first-only"],
                        ["hvg-encoder-budget", "--require-within-budget"]]


@POSIX
@pytest.mark.parametrize("updates", [{"MOCK_HVG_FIRST_MISSING": "1"}, {"MOCK_ENCODER_BUDGET_FAIL": "1"},
                                    {"MOCK_PANELS_MISSING": "1"}])
def test_encoder_failures_never_submit_arrays(pending_cluster, updates):
    result = invoke(pending_cluster, "encoders", **updates)
    assert result.returncode != 0
    assert submitted(pending_cluster) == []


@POSIX
def test_encoder_configuration_must_map_exactly_twenty_tasks(pending_cluster):
    path = pending_cluster[0] / "data/splits/cell_kfold.json"
    path.write_text(json.dumps({"folds": [{}] * 9}))
    result = invoke(pending_cluster, "encoder-pilot", MOCK_HVG_FIRST_MISSING="1")
    assert result.returncode != 0
    assert "twenty saved fold tasks" in result.stderr
    assert submitted(pending_cluster) == []


@POSIX
def test_predictor_pilot_dependencies_cover_only_unfinished_fold_encoder_arrays(pending_cluster):
    save_ids(pending_cluster, ("hvg_encoder_genept", "301"), ("hvg_encoder_uce", "302"), ("hvg_encoder_nicheformer", "303"))
    result = invoke(pending_cluster, "pilot", MOCK_HVG_MISSING_ENCODERS="uce,nicheformer", MOCK_ACTIVE_IDS="302,303")
    assert result.returncode == 0, result.stderr
    calls = submitted(pending_cluster)
    assert "--dependency=afterok:302:303" in calls[0]["args"]
    assert "--dependency=afterok:101" in calls[1]["args"]
    assert "--dependency=afterok:102" in calls[2]["args"]
    assert [call["args"][call["args"].index("--targets") + 1] for call in calls] == ["17"] * 3
    assert [len(call["args"][call["args"].index("--representations") + 1:call["args"].index("--targets")]) for call in calls] == [8, 11, 11]
    assert all(call["args"][-1] == "results/pending_workflow/pilot_with_hvg_encoders/timings.json" for call in calls)


@POSIX
def test_inactive_missing_fold_embeddings_prevent_predictor_pilots(pending_cluster):
    save_ids(pending_cluster, ("hvg_encoder_uce", "302"))
    result = invoke(pending_cluster, "pilot", MOCK_HVG_MISSING_ENCODERS="uce")
    assert result.returncode != 0
    assert "No valid artifact or active hvg_encoder_uce job" in result.stderr
    assert submitted(pending_cluster) == []


@POSIX
def test_standalone_encoder_profile_has_180_tasks_and_no_full_embedding_requirement(pending_cluster):
    state = pending_cluster[0] / "results/hvg_embeddings_workflow"
    state.mkdir()
    (state / "baseline-snapshot.json").write_text("{}")
    result = invoke(pending_cluster, "pilot", script="hvg_embeddings.sh", MOCK_EMBEDDINGS_MISSING="1")
    assert result.returncode == 0, result.stderr
    assert not any("--embedding" in call["args"] for call in records(pending_cluster))
    calls = submitted(pending_cluster)
    assert len(calls) == 3
    for call in calls:
        assert call["args"][call["args"].index("--representations") + 1:call["args"].index("--targets")] == list(HVG_INPUTS)
    # Fresh fixture accounting is inactive, so a preview may follow without mutations.
    result = invoke(pending_cluster, "submit", script="hvg_embeddings.sh")
    assert result.returncode == 0, result.stderr
    assert "0-179%4" in result.stdout
    assert "target/fold fits: 3060" in result.stdout


@POSIX
@pytest.mark.parametrize("stage", ["panels", "encoder-pilot", "encoders", "pilot", "submit"])
def test_every_new_mutating_stage_guards_all_three_workflows(pending_cluster, stage):
    state = pending_cluster[0] / "results/hvg_embeddings_workflow"
    state.mkdir()
    (state / "slurm-jobs.tsv").write_text("hvg_encoder_uce\t999\n")
    result = invoke(pending_cluster, stage, MOCK_ACTIVE_IDS="999")
    assert result.returncode == 2
    assert "still queued/running" in result.stderr
    assert not any(call["command"] in {"run", "bootstrap", "sbatch"} for call in records(pending_cluster))


@POSIX
def test_delayed_report_rejects_more_than_the_declared_600_tasks(pending_cluster):
    save_ids(pending_cluster, ("production", "1001"))
    result = invoke(pending_cluster, "report", MOCK_ACCOUNTING_SUCCESS="1", MOCK_ACCOUNTING_TASKS="601")
    assert result.returncode != 0
    assert "all 600 tasks" in result.stderr
    assert submitted(pending_cluster) == []


def test_encoder_and_predictor_resource_requests_remain_separate():
    root = Path(__file__).parents[1]
    encoder = (root / "cluster/hvg_encoder_array.sbatch").read_text()
    for resource in ("--cpus-per-task=16", "--mem=64G", "--gres=gpu:1", "--time=04:00:00"):
        assert resource in encoder
    assert "--gene-set hvg" in encoder and "--protocol" in encoder and "--fold" in encoder
    production = (root / "cluster/evaluate.sbatch").read_text()
    for resource in ("--cpus-per-task=8", "--mem=32G", "--gres=gpu:1", "--time=06:00:00"):
        assert resource in production


@POSIX
def test_valid_artifacts_still_depend_on_active_encoder_writers(pending_cluster):
    save_ids(pending_cluster, ("hvg_encoder_genept", "301"), ("hvg_encoder_uce", "302"), ("hvg_encoder_nicheformer", "303"))
    result = invoke(pending_cluster, "pilot", MOCK_ACTIVE_IDS="301,302,303")
    assert result.returncode == 0, result.stderr
    assert "--dependency=afterok:301:302:303" in submitted(pending_cluster)[0]["args"]


@POSIX
def test_active_global_encoder_prevents_gpu_fold_sweep(pending_cluster):
    save_ids(pending_cluster, ("uce", "401"))
    result = invoke(pending_cluster, "encoders", MOCK_ACTIVE_IDS="401")
    assert result.returncode == 2
    assert "still queued/running" in result.stderr
    assert submitted(pending_cluster) == []
