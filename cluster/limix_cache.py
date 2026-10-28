#!/usr/bin/env python3
"""Redirect only the frozen LimiX vendor's machine-specific cache directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

RELATIVE = "inference/v2_0/predictor.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))
OLD = b'self.cache_manager = self.CacheManager(cache_dir="/mnt/public/lijiansheng0830cache")'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _git_source(repo):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], timeout=30).decode().strip()
    source = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{RELATIVE}"], timeout=30)
    return commit, source


def _active_project_jobs(root):
    # Apply before submitting pilots, never while a project worker could hash
    # or import the vendor file. Pending jobs can stay queued.
    result = subprocess.run(["squeue", "-h", "--me", "-o", "%i|%T|%Z|%j"],
                            check=True, capture_output=True, text=True, timeout=30)
    active = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise RuntimeError("Could not interpret the project queue; patch was not applied")
        job, state, workdir, name = (item.strip() for item in fields)
        if state != "PENDING" and Path(workdir).resolve() == root and name in {"cellpilot", "cellpred", "cellpredict"}:
            active.append(job)
    return active


def _existing_limix_results(config, root):
    from cell_property_prediction.config import resolve

    results = resolve(root, config["paths"].get("results_dir", "results"))
    found = list(results.rglob("limix/fold_*/manifest.json"))
    found.extend(results.rglob("limix/fold_*/targets/*.json"))
    for path in results.rglob("timings.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if any(row.get("predictor") == "limix" for row in value.get("runs", [])):
            found.append(path)
    return sorted(set(found))


def _write(path, data, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".cpp-{uuid.uuid4().hex[:12]}.tmp"
    try:
        temporary.write_bytes(data)
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure(config, root, *, apply=False):
    from cell_property_prediction.config import resolve
    from login_hvg import _lock

    root = Path(root).resolve()
    repo = resolve(root, config["models"]["limix"]["repo_dir"]).resolve()
    cache_root = resolve(root, config["paths"].get("cache_dir", "cache")).resolve()
    cache = cache_root / "limix_native"
    if cache.is_relative_to(repo):
        raise RuntimeError("The native cache must lie outside the hashed LimiX vendor directory")
    source = repo / RELATIVE
    manifest_path = cache_root / "model_manifest.json"
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))["limix"]["vendor"]
    commit, original = _git_source(repo)
    if frozen.get("commit") != commit or resolve(root, frozen["path"]).resolve() != repo:
        raise RuntimeError("LimiX vendor HEAD/path differs from the frozen model manifest")
    if original.count(OLD) != 1:
        raise RuntimeError("Unsupported frozen LimiX cache constructor; no source was changed")
    replacement = ("self.cache_manager = self.CacheManager(cache_dir=" + repr(str(cache)) + ")").encode("utf-8")
    patched = original.replace(OLD, replacement, 1)
    current = source.read_bytes()
    if current not in (original, patched):
        raise RuntimeError("LimiX predictor has unrelated edits; no source was changed")
    directory = root / "logs/runtime_patches"
    record_path = directory / "limix-native-cache.json"
    record = {"schema_version": 1, "kind": "limix_cache_directory_only", "root": str(root),
              "vendor_commit": commit, "source": str(source), "cache_dir": str(cache),
              "original_sha256": _sha(original), "patched_sha256": _sha(patched),
              "original_cache": "/mnt/public/lijiansheng0830cache",
              "policy": "Replace only the cache-directory literal; native inference settings unchanged"}
    result = dict(record, status="verified" if current == patched else "patch_required")
    if not apply:
        return result
    # No scientific sources/model manifest are modified, and records/backups
    # stay outside vendor so they do not become additional model inputs.
    with _lock(directory / "limix-native-cache.lock"):
        current = source.read_bytes()
        if current not in (original, patched):
            raise RuntimeError("LimiX predictor changed while preparing the cache patch")
        if current == original:
            prior_results = _existing_limix_results(config, root)
            if prior_results:
                raise RuntimeError(f"Existing LimiX predictions/timings require provenance review before patching: {prior_results[0]}")
            active = _active_project_jobs(root)
            if active:
                raise RuntimeError(f"Project jobs are active ({', '.join(active)}); wait before patching the vendor")
        if record_path.exists():
            previous = json.loads(record_path.read_text(encoding="utf-8"))
            if any(previous.get(key) != value for key, value in record.items()):
                raise RuntimeError("Existing cache-patch provenance differs; inspect it before changing vendor files")
        backup = directory / f"limix-predictor-{_sha(original)[:16]}.original"
        diff = directory / "limix-native-cache.diff"
        difference = "".join(difflib.unified_diff(original.decode("utf-8").splitlines(keepends=True),
                                                   patched.decode("utf-8").splitlines(keepends=True),
                                                   fromfile=RELATIVE + ".original", tofile=RELATIVE))
        for path, data in ((backup, original), (diff, difference.encode("utf-8"))):
            if path.exists() and path.read_bytes() != data:
                raise RuntimeError(f"Existing patch audit file differs: {path}")
            if not path.exists():
                _write(path, data)
        cache.mkdir(parents=True, exist_ok=True)
        if current == original:
            _write(source, patched, source.stat().st_mode & 0o777)
        if not record_path.exists():
            _write(record_path, (json.dumps(dict(record, applied_at=datetime.now(timezone.utc).isoformat()),
                                            indent=2) + "\n").encode("utf-8"))
        if source.read_bytes() != patched:
            raise RuntimeError("LimiX vendor cache patch verification failed")
    return dict(record, status="verified", audit=str(record_path))


def main():
    from cell_property_prediction.config import load_config, resolve

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=os.environ.get("CPP_ROOT", "."))
    parser.add_argument("--config", default=os.environ.get("CPP_CONFIG", "configs/cluster.local.yaml"))
    parser.add_argument("--apply", action="store_true", help="Apply the one-literal patch and record provenance")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        result = ensure(load_config(resolve(root, args.config)), root, apply=args.apply)
    except (RuntimeError, ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
