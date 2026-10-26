"""The HVG extension is additive to the frozen original experiment matrix."""
import json
from pathlib import Path

import pytest

from cell_property_prediction.cli import parser
from cell_property_prediction.config import atomic_json, load_config
from cell_property_prediction.jobs import compute_fingerprint, create_jobs, estimate_jobs


def test_hvg_stage_has_both_inputs_all_models_and_existing_folds(tmp_path):
    config_path = Path(__file__).parents[1] / "configs/benchmark.yaml"
    before = config_path.read_bytes()
    config = load_config(config_path)
    for protocol in config["evaluation"]["protocols"]:
        atomic_json(tmp_path / config["paths"]["splits_dir"] / f"{protocol}.json", {"folds": [{}] * 10})
    output = tmp_path / "jobs/hvg.tsv"
    jobs = create_jobs(config, tmp_path, "hvg", output)
    assert len(jobs) == 120
    expected = {(protocol, representation, predictor, fold)
                for protocol in ("cell_kfold", "mouse_group_kfold")
                for representation in ("hvg", "hvg_pca128")
                for predictor in ("tabpfn", "limix", "kumo") for fold in range(10)}
    assert {tuple(job.values()) for job in jobs} == expected
    assert json.loads(output.with_suffix(".json").read_text())["target_fits"] == 2040
    assert config_path.read_bytes() == before
    assert parser().parse_args(["jobs", "--stage", "hvg", "--output", str(output)]).stage == "hvg"
    assert not {job["representation"] for job in create_jobs(config, tmp_path, "pending", tmp_path / "pending.tsv")} & {"hvg", "hvg_pca128"}

    pairs = sorted({(job["representation"], job["predictor"]) for job in jobs})
    pilot_path = tmp_path / "results/hvg_workflow/pilot/timings.json"
    runs = [{"representation": representation, "predictor": predictor, "targets": 2,
             "elapsed_seconds": 10, "device": "cuda"} for representation, predictor in pairs]
    pilot = {"fingerprint": compute_fingerprint(config, tmp_path), "runs": runs}
    atomic_json(pilot_path, pilot)
    config["cluster"]["remaining_gpu_hours"] = 10
    estimate = estimate_jobs(config, jobs, pilot_path, root=tmp_path)
    assert estimate["jobs"] == 120
    assert estimate["within_budget"]
    assert estimate["estimated_gpu_hours_with_2x_margin"] == pytest.approx(120 * 2 * 10 * 17 / 2 / 3600)
    pilot["runs"] = runs[:-1]
    atomic_json(pilot_path, pilot)
    with pytest.raises(ValueError, match="Missing pilot measurements"):
        estimate_jobs(config, jobs, pilot_path, root=tmp_path)
