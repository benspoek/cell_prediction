"""Synthetic coverage for continuing the matrix after the simple TabPFN jobs."""
import json
from pathlib import Path

import pytest

from cell_property_prediction.cli import parser
from cell_property_prediction.config import atomic_json, load_config
from cell_property_prediction.jobs import create_jobs, compute_fingerprint, estimate_jobs


@pytest.fixture
def pending_config(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs/benchmark.yaml")
    for protocol in config["evaluation"]["protocols"]:
        atomic_json(tmp_path / config["paths"]["splits_dir"] / f"{protocol}.json",
                    {"folds": [{}] * 10})
    return config


def job_key(job):
    return tuple(job[field] for field in ("protocol", "representation", "predictor", "fold"))


def test_pending_matrix_is_full_matrix_without_simple_jobs(tmp_path, pending_config):
    pending = create_jobs(pending_config, tmp_path, "pending", tmp_path / "pending.tsv")
    full = create_jobs(pending_config, tmp_path, "full", tmp_path / "full.tsv")
    simple = create_jobs(pending_config, tmp_path, "simple", tmp_path / "simple.tsv")
    pending_keys, full_keys, simple_keys = map(lambda rows: set(map(job_key, rows)),
                                              (pending, full, simple))

    assert len(simple) == 60
    assert len(full) == 480
    assert len(pending) == len(pending_keys) == 420
    assert pending_keys == full_keys - simple_keys
    assert not pending_keys & simple_keys
    pairs = {(job["representation"], job["predictor"]) for job in pending}
    assert len(pairs) == 21
    assert len({(job["representation"], job["predictor"]) for job in full}) == 24
    for predictor in ("tabpfn", "limix", "kumo"):
        assert {(name, predictor) for name in ("hvg", "hvg_pca128")} <= pairs
    assert len({name for name, predictor in pairs if predictor == "limix"}) == 8
    assert len({name for name, predictor in pairs if predictor == "kumo"}) == 8
    assert {job["predictor"] for job in pending} == {"tabpfn", "limix", "kumo"}
    assert {job["representation"] for job in pending if job["predictor"] == "tabpfn"} == {
        "genept", "uce", "nicheformer", "hvg", "hvg_pca128"
    }
    assert json.loads((tmp_path / "pending.json").read_text()) == {
        "stage": "pending", "jobs": 420, "targets_per_job": 17, "target_fits": 7140,
        "fields": ["protocol", "representation", "predictor", "fold"],
    }
    assert parser().parse_args(["jobs", "--stage", "pending", "--output", "pending.tsv"]).stage == "pending"
    assert parser().parse_args(["pilot", "--predictors", "tabpfn"]).representations == [
        "pca128", "ion_channels", "authors25", "genept", "uce", "nicheformer", "hvg", "hvg_pca128"
    ]


def test_pending_budget_requires_only_remaining_pairs(tmp_path, pending_config):
    jobs = create_jobs(pending_config, tmp_path, "pending", tmp_path / "pending.tsv")
    pairs = sorted({(job["representation"], job["predictor"]) for job in jobs})
    pilot_path = tmp_path / "results/pending_workflow/pilot/timings.json"
    runs = [{"representation": representation, "predictor": predictor, "targets": 2,
             "elapsed_seconds": 10, "device": "cuda"} for representation, predictor in pairs]
    pilot = {"fingerprint": compute_fingerprint(pending_config, tmp_path), "runs": runs}
    atomic_json(pilot_path, pilot)
    pending_config["cluster"]["remaining_gpu_hours"] = 30
    estimate = estimate_jobs(pending_config, jobs, pilot_path, root=tmp_path)
    assert estimate["jobs"] == 420
    assert estimate["estimated_gpu_hours_with_2x_margin"] == pytest.approx(420 * 2 * 10 * 17 / 2 / 3600)
    assert estimate["within_budget"]
    pending_config["cluster"]["remaining_gpu_hours"] = 15
    assert not estimate_jobs(pending_config, jobs, pilot_path, root=tmp_path)["within_budget"]

    pilot["runs"] = [row for row in runs if row["representation"] not in {"hvg", "hvg_pca128"}]
    atomic_json(pilot_path, pilot)
    with pytest.raises(ValueError, match="Missing pilot measurements.*hvg"):
        estimate_jobs(pending_config, jobs, pilot_path, root=tmp_path)

    pilot["runs"] = runs[:-1]
    atomic_json(pilot_path, pilot)
    with pytest.raises(ValueError, match="Missing pilot measurements"):
        estimate_jobs(pending_config, jobs, pilot_path, root=tmp_path)


@pytest.mark.parametrize("stage", ["full", "pending"])
def test_combined_matrix_deduplicates_already_configured_hvg_inputs(tmp_path, pending_config, stage):
    pending_config["evaluation"]["representations"].extend(["hvg", "hvg_pca128", "hvg"])
    jobs = create_jobs(pending_config, tmp_path, stage, tmp_path / "jobs.tsv")
    keys = list(map(job_key, jobs))
    assert len(keys) == len(set(keys)) == (480 if stage == "full" else 420)
    assert pending_config["evaluation"]["representations"][-3:] == ["hvg", "hvg_pca128", "hvg"]
