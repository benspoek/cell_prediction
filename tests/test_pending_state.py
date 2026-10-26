"""Synthetic continuation checks: no real datasets, checkpoints or model inference."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from cell_property_prediction.config import atomic_json, file_sha256, load_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pending_state", ROOT / "cluster/pending_state.py")
state = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(state)


@pytest.fixture
def baseline(tmp_path):
    config = load_config(ROOT / "configs/benchmark.yaml")
    config["data"]["expected_cohort_cells"] = 4
    config["evaluation"]["n_splits"] = 2
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    prepared = tmp_path / "data/processed"
    prepared.mkdir(parents=True)
    for name in ("manifest.json", "expression.npz", "counts.h5ad", "targets_author.csv"):
        (prepared / name).write_text("synthetic input")
    targets = ["AP amplitude", "AP width"]
    cells = ["cell-a", "cell-b", "cell-c", "cell-d"]
    (prepared / "targets_raw.csv").write_text(
        "Cell,AP amplitude,AP width\ncell-a,1,2\ncell-b,2,3\ncell-c,3,4\ncell-d,4,5\n")
    (prepared / "metadata.csv").write_text(
        "Cell,Mouse\ncell-a,m1\ncell-b,m1\ncell-c,m2\ncell-d,m2\n")
    model = tmp_path / "cache/models/tabpfn/model.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"synthetic checkpoint")
    atomic_json(tmp_path / "cache/model_manifest.json", {"tabpfn": {"revision": "fixture"}})
    atomic_json(tmp_path / "results/pilot/timings.json", {"original": True})
    reports = tmp_path / "results/reports"
    reports.mkdir()
    (reports / "summary.csv").write_text("old summary\n")
    common = {str(path): file_sha256(path) for path in prepared.iterdir()}
    for protocol in state.PROTOCOLS:
        folds = [
            {"train_ids": cells[2:], "test_ids": cells[:2]},
            {"train_ids": cells[:2], "test_ids": cells[2:]},
        ]
        split = tmp_path / "data/splits" / f"{protocol}.json"
        atomic_json(split, {"folds": folds})
        inputs = {**common, str(split): file_sha256(split)}
        for representation in state.SIMPLE:
            for index, fold in enumerate(folds):
                directory = tmp_path / "results" / protocol / representation / "tabpfn" / f"fold_{index:02d}"
                directory.mkdir(parents=True)
                prediction = directory / "predictions.csv"
                with prediction.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=[
                        "cell_id", "target", "y_pred", "protocol", "representation", "predictor", "fold"])
                    writer.writeheader()
                    for cell in fold["test_ids"]:
                        for target in targets:
                            writer.writerow({"cell_id": cell, "target": target, "y_pred": 1,
                                "protocol": protocol, "representation": representation,
                                "predictor": "tabpfn", "fold": index})
                atomic_json(directory / "targets/synthetic-target.json", {"original": True})
                atomic_json(directory / "manifest.json", {
                    "status": "complete", "protocol": protocol, "representation": representation,
                    "predictor": "tabpfn", "fold": index, "target_names": targets,
                    "pilot_target_subset": False, "config_sha256": state._config_hash(config),
                    "input_sha256": inputs, "model_input_sha256": {str(model): file_sha256(model)},
                    "code_sha256": {"models.py": "old-source-version"},
                    "targets": [{"target": t, "status": "complete"} for t in targets],
                    "prediction_sha256": file_sha256(prediction),
                })
    snapshot = tmp_path / "results/pending_workflow/baseline-snapshot.json"
    return tmp_path, config, config_path, snapshot


def test_snapshot_preserves_baselines_while_models_and_reports_expand(baseline):
    root, config, path, snapshot = baseline
    assert state.snapshot(config, root, path, snapshot)["baseline_folds"] == 12
    original = snapshot.read_bytes()
    atomic_json(root / "cache/model_manifest.json", {
        "tabpfn": {"revision": "fixture"}, "kumo": {"revision": "new-pending-model"}})
    (root / "results/reports/summary.csv").write_text("expanded summary\n")
    config["cluster"]["remaining_gpu_hours"] = 630
    path.write_text(yaml.safe_dump(config))
    assert state.check(config, root, snapshot)["status"] == "verified"
    assert state.snapshot(config, root, path, snapshot)["status"] == "verified"
    assert snapshot.read_bytes() == original
    archive = snapshot.with_suffix("")
    assert (archive / "reports/summary.csv").read_text() == "old summary\n"
    assert json.loads((archive / "model_manifest.json").read_text()) == {"tabpfn": {"revision": "fixture"}}


def test_snapshot_checks_target_level_artifacts_and_compute_configuration(baseline):
    root, config, path, snapshot = baseline
    state.snapshot(config, root, path, snapshot)
    altered_config = copy.deepcopy(config)
    altered_config["models"]["tabpfn"]["n_estimators"] = 4
    with pytest.raises(ValueError, match="configuration"):
        state.check(altered_config, root, snapshot)
    target = next((root / "results/cell_kfold/authors25/tabpfn").glob("fold_*/targets/*.json"))
    target.write_text("{}")
    with pytest.raises(ValueError, match="Protected baseline"):
        state.check(config, root, snapshot)


def test_snapshot_rejects_partial_baseline_or_modified_original_model(baseline):
    root, config, path, snapshot = baseline
    manifest_path = root / "results/cell_kfold/authors25/tabpfn/fold_00/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["targets"][0]["status"] = "failed"
    atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="completed target"):
        state.snapshot(config, root, path, snapshot)
    manifest["targets"][0]["status"] = "complete"
    atomic_json(manifest_path, manifest)
    state.snapshot(config, root, path, snapshot)
    atomic_json(root / "cache/model_manifest.json", {"tabpfn": {"revision": "changed"}})
    with pytest.raises(ValueError, match="TabPFN resource"):
        state.check(config, root, snapshot)


def _embedding(root, config, name):
    path = root / "embeddings" / f"{name}.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez(path, values=np.ones((4, state.EMBEDDINGS[name]), dtype=np.float32),
             cell_ids=np.array(["cell-a", "cell-b", "cell-c", "cell-d"]))
    prepared_sha = file_sha256(root / "data/processed/counts.h5ad")
    context = {"embedding": name, "config": config["embeddings"][name], "prepared_sha256": prepared_sha}
    context_hash = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_json(path.with_suffix(".audit.json"), {
        "output_sha256": file_sha256(path), "labels_used": False, "frozen": True,
        "prepared_sha256": prepared_sha, "context": context, "context_sha256": context_hash,
    })
    return path


def test_embedding_readiness_handles_partial_completion_and_provenance(baseline):
    root, config, path, snapshot = baseline
    state.snapshot(config, root, path, snapshot)
    _embedding(root, config, "genept")
    assert state.check(config, root, snapshot, names=["genept"])["embeddings"] == {"genept": 3072}
    with pytest.raises(ValueError, match="uce"):
        state.check(config, root, snapshot, embeddings=True)
    for name in ("uce", "nicheformer"):
        _embedding(root, config, name)
    assert len(state.check(config, root, snapshot, embeddings=True)["embeddings"]) == 3
    audit_path = root / "embeddings/genept.audit.json"
    audit = json.loads(audit_path.read_text())
    audit["prepared_sha256"] = "another-prepared-input"
    atomic_json(audit_path, audit)
    with pytest.raises(ValueError, match="provenance"):
        state.check(config, root, snapshot, names=["genept"])


def test_embedding_readiness_rejects_wrong_cell_ids(baseline):
    root, config, path, snapshot = baseline
    state.snapshot(config, root, path, snapshot)
    output = _embedding(root, config, "genept")
    np.savez(output, values=np.ones((4, 3072)), cell_ids=np.array(["cell-a"] * 4))
    audit_path = output.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    audit["output_sha256"] = file_sha256(output)
    atomic_json(audit_path, audit)
    with pytest.raises(ValueError, match="dimensions, IDs"):
        state.check(config, root, snapshot, names=["genept"])
