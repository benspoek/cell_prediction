"""Vendor-only cache fix, frozen provenance and synthetic constructor regression."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from cell_property_prediction.evaluation import _model_file_inputs
from cell_property_prediction.jobs import compute_fingerprint

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cluster"))
import login_hvg
SPEC = importlib.util.spec_from_file_location("limix_cache_test", REPO / "cluster/limix_cache.py")
fix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fix)

SOURCE = b'''import os
class LimiXPredictor:
    class CacheManager:
        def __init__(self, cache_dir):
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
    def __init__(self, use_data_cache=False):
        self.cache_manager = self.CacheManager(cache_dir="/mnt/public/lijiansheng0830cache")
        self.use_data_cache = use_data_cache
'''


@pytest.fixture
def rig(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "project with spaces"
    repo = root / "vendor/LimiX"
    source = repo / fix.RELATIVE
    source.parent.mkdir(parents=True)
    source.write_bytes(SOURCE)
    cache = root / "cache"
    cache.mkdir()
    manifest = cache / "model_manifest.json"
    commit = "a" * 40
    manifest.write_text(json.dumps({"limix": {"vendor": {"path": str(repo), "commit": commit}}}))
    config = {"paths": {"cache_dir": "cache", "prepared_dir": "data/processed", "splits_dir": "data/splits",
                        "embeddings_dir": "embeddings", "results_dir": "results"},
              "models": {"limix": {"repo_dir": "vendor/LimiX"}, "tabpfn": {}}, "cluster": {}}
    monkeypatch.setattr(fix, "_git_source", lambda path: (commit, SOURCE))
    monkeypatch.setattr(fix, "_active_project_jobs", lambda path: [])
    monkeypatch.setattr(login_hvg, "_lock", lambda path: nullcontext())
    return SimpleNamespace(root=root, repo=repo, source=source, cache=cache, manifest=manifest,
                           config=config, commit=commit, audit=root / "logs/runtime_patches/limix-native-cache.json")


def test_preview_does_not_change_vendor_or_create_audit(rig):
    before = rig.manifest.read_bytes()
    result = fix.ensure(rig.config, rig.root)
    assert result["status"] == "patch_required"
    assert rig.source.read_bytes() == SOURCE and rig.manifest.read_bytes() == before
    assert not rig.audit.exists() and not (rig.cache / "limix_native").exists()


def test_literal_fix_runs_constructor_without_changing_disabled_cache_setting(rig):
    result = fix.ensure(rig.config, rig.root, apply=True)
    expected_cache = rig.cache / "limix_native"
    expected = SOURCE.replace(fix.OLD, ("self.cache_manager = self.CacheManager(cache_dir=" + repr(str(expected_cache)) + ")").encode())
    assert rig.source.read_bytes() == expected
    namespace = {}
    exec(rig.source.read_bytes(), namespace)
    model = namespace["LimiXPredictor"](use_data_cache=False)
    assert model.use_data_cache is False and model.cache_manager.cache_dir == str(expected_cache)
    audit = json.loads(rig.audit.read_text())
    assert result["status"] == "verified" and audit["vendor_commit"] == rig.commit
    assert audit["original_sha256"] == hashlib.sha256(SOURCE).hexdigest()
    assert audit["patched_sha256"] == hashlib.sha256(expected).hexdigest()
    backup = list(rig.audit.parent.glob("*.original"))
    assert len(backup) == 1 and backup[0].read_bytes() == SOURCE
    assert not list(rig.repo.rglob("*.original"))


def test_common_pilot_and_tabpfn_provenance_are_preserved_limix_records_new_bytes(rig):
    common = compute_fingerprint(rig.config, rig.root)
    tabpfn = _model_file_inputs(rig.root, rig.config["models"]["tabpfn"])
    limix = _model_file_inputs(rig.root, rig.config["models"]["limix"])
    manifest = rig.manifest.read_bytes()
    fix.ensure(rig.config, rig.root, apply=True)
    assert compute_fingerprint(rig.config, rig.root) == common
    assert rig.manifest.read_bytes() == manifest
    assert _model_file_inputs(rig.root, rig.config["models"]["tabpfn"]) == tabpfn
    changed = _model_file_inputs(rig.root, rig.config["models"]["limix"])
    assert set(changed) == set(limix) and changed != limix


def test_idempotent_verification_is_safe_while_other_jobs_run(rig, monkeypatch):
    fix.ensure(rig.config, rig.root, apply=True)
    before = rig.source.read_bytes(), rig.audit.read_bytes()
    def forbidden(*args):
        raise AssertionError("Already-patched source needs no scheduler mutation guard")
    monkeypatch.setattr(fix, "_active_project_jobs", forbidden)
    assert fix.ensure(rig.config, rig.root, apply=True)["status"] == "verified"
    assert before == (rig.source.read_bytes(), rig.audit.read_bytes())


def test_interrupted_record_write_can_be_recovered_from_exact_vendor_bytes(rig):
    fix.ensure(rig.config, rig.root, apply=True)
    rig.audit.unlink()
    assert fix.ensure(rig.config, rig.root, apply=True)["status"] == "verified"
    assert rig.audit.is_file()


@pytest.mark.parametrize("edit", [b"# unexpected change\n", b"# extra\n"])
def test_unrelated_vendor_edits_are_not_overwritten(rig, edit):
    changed = SOURCE + edit
    rig.source.write_bytes(changed)
    with pytest.raises(RuntimeError, match="unrelated edits"):
        fix.ensure(rig.config, rig.root, apply=True)
    assert rig.source.read_bytes() == changed


def test_vendor_revision_mismatch_stops_before_changes(rig, monkeypatch):
    monkeypatch.setattr(fix, "_git_source", lambda path: ("b"*40, SOURCE))
    with pytest.raises(RuntimeError, match="frozen model manifest"):
        fix.ensure(rig.config, rig.root, apply=True)
    assert rig.source.read_bytes() == SOURCE and not rig.audit.exists()


@pytest.mark.parametrize("original", [SOURCE.replace(fix.OLD, b"pass"), SOURCE + fix.OLD])
def test_unknown_or_ambiguous_upstream_constructor_fails_closed(rig, monkeypatch, original):
    monkeypatch.setattr(fix, "_git_source", lambda path: (rig.commit, original))
    with pytest.raises(RuntimeError, match="Unsupported frozen"):
        fix.ensure(rig.config, rig.root, apply=True)
    assert rig.source.read_bytes() == SOURCE


def test_running_predictor_stops_first_patch(rig, monkeypatch):
    monkeypatch.setattr(fix, "_active_project_jobs", lambda path: ["5253941"])
    with pytest.raises(RuntimeError, match="Project jobs are active"):
        fix.ensure(rig.config, rig.root, apply=True)
    assert rig.source.read_bytes() == SOURCE and not rig.audit.exists()


@pytest.mark.parametrize("relative,payload", [
    ("cell_kfold/pca128/limix/fold_00/manifest.json", "{}"),
    ("cell_kfold/pca128/limix/fold_00/targets/target.json", "{}"),
    ("pending_workflow/pilot/timings.json", '{"runs":[{"predictor":"limix"}]}'),
])
def test_existing_limix_outputs_need_review_instead_of_silent_reuse(rig, relative, payload):
    path = rig.root / "results" / relative
    path.parent.mkdir(parents=True)
    path.write_text(payload)
    with pytest.raises(RuntimeError, match="Existing LimiX predictions"):
        fix.ensure(rig.config, rig.root, apply=True)
    assert rig.source.read_bytes() == SOURCE


def test_completed_tabpfn_timings_are_kept(rig):
    path = rig.root / "results/pending_workflow/pilot/timings.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"runs":[{"predictor":"tabpfn"}]}')
    before = path.read_bytes()
    fix.ensure(rig.config, rig.root, apply=True)
    assert path.read_bytes() == before


def test_altered_audit_is_rejected(rig):
    fix.ensure(rig.config, rig.root, apply=True)
    data = json.loads(rig.audit.read_text())
    data["patched_sha256"] = "wrong"
    rig.audit.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="provenance differs"):
        fix.ensure(rig.config, rig.root, apply=True)


def test_queue_guard_allows_encoders_and_pending_predictors_but_blocks_running_predictors(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    output = (f"1|RUNNING|{root}|cellhvgencoder\n"
              f"2|PENDING|{root}|cellpilot\n"
              f"3|RUNNING|{root}|cellpilot\n"
              f"4|RUNNING|{root.parent / 'other'}|cellpred\n"
              f"5|RUNNING|{root}|cellpredict\n")
    monkeypatch.setattr(fix.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=output))
    assert fix._active_project_jobs(root) == ["3", "5"]
