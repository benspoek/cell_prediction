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
    assert len(full) == 360
    assert len(pending) == len(pending_keys) == 300
    assert pending_keys == full_keys - simple_keys
    assert not pending_keys & simple_keys
    pairs = {(job["representation"], job["predictor"]) for job in pending}
    assert len(pairs) == 15
    assert {job["predictor"] for job in pending} == {"tabpfn", "limix", "kumo"}
    assert {job["representation"] for job in pending if job["predictor"] == "tabpfn"} == {
        "genept", "uce", "nicheformer"
    }
    assert json.loads((tmp_path / "pending.json").read_text()) == {
        "stage": "pending", "jobs": 300, "targets_per_job": 17, "target_fits": 5100,
        "fields": ["protocol", "representation", "predictor", "fold"],
    }
    assert parser().parse_args(["jobs", "--stage", "pending", "--output", "pending.tsv"]).stage == "pending"


def test_pending_budget_requires_only_remaining_pairs(tmp_path, pending_config):
    jobs = create_jobs(pending_config, tmp_path, "pending", tmp_path / "pending.tsv")
    pairs = sorted({(job["representation"], job["predictor"]) for job in jobs})
    pilot_path = tmp_path / "results/pending_workflow/pilot/timings.json"
    runs = [{"representation": representation, "predictor": predictor, "targets": 2,
             "elapsed_seconds": 10, "device": "cuda"} for representation, predictor in pairs]
    pilot = {"fingerprint": compute_fingerprint(pending_config, tmp_path), "runs": runs}
    atomic_json(pilot_path, pilot)
    pending_config["cluster"]["remaining_gpu_hours"] = 15
    estimate = estimate_jobs(pending_config, jobs, pilot_path, root=tmp_path)
    assert estimate["jobs"] == 300
    assert estimate["estimated_gpu_hours_with_2x_margin"] == pytest.approx(300 * 2 * 10 * 17 / 2 / 3600)
    assert estimate["within_budget"]

    pilot["runs"] = runs[:-1]
    atomic_json(pilot_path, pilot)
    with pytest.raises(ValueError, match="Missing pilot measurements"):
        estimate_jobs(pending_config, jobs, pilot_path, root=tmp_path)
