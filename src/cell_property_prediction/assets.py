"""Explicit, resumable downloads. Run this module on the cluster, never inside a fit."""
from __future__ import annotations

import csv
import hashlib
import http.client
import io
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import atomic_json, file_sha256, resolve

SOURCE_FILES = (
    "m1_patchseq_exon_counts.csv.gz", "m1_patchseq_intron_counts.csv.gz",
    "m1_patchseq_meta_data.csv", "m1_patchseq_ephys_features.csv",
    "gene_lengths.txt", "ion-channel-genes-group-177.csv",
)


# Published UCE files, Figshare article 24320806 v5. Pin metadata so an empty,
# truncated or HTML response cannot become a model resource. Download-only code
# lives here to preserve the frozen encoder implementation fingerprints.
# https://api.figshare.com/v2/articles/24320806
_FIGSHARE_FILES = {
    "42706555": (139, "ddf9143508857e62607f323b338ff383"),
    "42706558": (4097781, "42a9b5871b1ed955fd05414b3ab89081"),
    "42706585": (2979205876, "3523ea431c403f95701b6eeb0c30b997"),
    "42715213": (2735410523, "6756d7fa8348e0f70b5e5b11e324f023"),
    "43423236": (5686283829, "eed62d4fe51873c87bcf9660faa07707"),
}


def _md5(path: Path) -> str:
    # MD5 verifies the publisher's checksum; provenance also records SHA256.
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> dict:
    """Reuse valid assets; atomically publish only complete, nonempty downloads."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlsplit(url)
    prefix = {"figshare.com": "/ndownloader/files/",
              "ndownloader.figshare.com": "/files/"}.get(parsed.hostname)
    file_id = parsed.path[len(prefix):] if prefix and parsed.path.startswith(prefix) else None
    expected = _FIGSHARE_FILES.get(file_id)
    download_url = f"https://ndownloader.figshare.com/files/{file_id}" if expected else url

    def valid_cache() -> bool:
        if not destination.is_file() or destination.stat().st_size == 0:
            return False
        return expected is None or (destination.stat().st_size == expected[0]
                                    and _md5(destination) == expected[1])

    if not valid_cache():
        temporary = destination.with_name(destination.name + ".download")
        for attempt in range(3):
            try:
                request = urllib.request.Request(download_url, headers={
                    "User-Agent": "cell-property-prediction/0.1", "Accept-Encoding": "identity"})
                with urllib.request.urlopen(request, timeout=180) as response:
                    status = getattr(response, "status", None)
                    if status is not None and status != 200:
                        raise ValueError(f"unexpected HTTP status {status}")
                    headers = getattr(response, "headers", {})
                    content_length = headers.get("Content-Length")
                    declared_size = int(content_length) if content_length is not None else None
                    if declared_size is not None and (declared_size <= 0 or
                            (expected is not None and declared_size != expected[0])):
                        raise ValueError(f"unexpected Content-Length {declared_size}; "
                                         f"expected {expected[0] if expected else 'a nonempty response'}")
                    digest = hashlib.md5()
                    received = 0
                    with temporary.open("wb") as output:
                        while block := response.read(1024 * 1024):
                            output.write(block)
                            digest.update(block)
                            received += len(block)
                    if received == 0:
                        raise ValueError("empty response")
                    if declared_size is not None and received != declared_size:
                        raise ValueError(f"incomplete response: {received} of {declared_size} bytes")
                    if expected is not None:
                        if received != expected[0]:
                            raise ValueError(f"incorrect file size: {received}; expected {expected[0]} bytes")
                        if digest.hexdigest() != expected[1]:
                            raise ValueError("file checksum differs from the published Figshare MD5")
                os.replace(temporary, destination)
                break
            except (OSError, urllib.error.URLError, http.client.HTTPException, ValueError) as error:
                if attempt == 2:
                    raise RuntimeError(f"Download failed for {destination} after 3 attempts: {error}. "
                                       "No invalid download was installed. Retry setup; valid cached "
                                       "assets will be reused.") from error
                time.sleep(2 ** attempt)
            finally:
                temporary.unlink(missing_ok=True)
    record = {"url": url, "path": str(destination), "sha256": file_sha256(destination),
              "bytes": destination.stat().st_size}
    if expected is not None:
        record.update(download_url=download_url, expected_bytes=expected[0], published_md5=expected[1])
    return record


def fetch_data(config: dict, root: Path) -> dict:
    raw_dir = resolve(root, config["paths"]["raw_dir"])
    commit = config["data"]["source_commit"]
    repository = config["data"]["source_repo"]
    files = [download(f"https://raw.githubusercontent.com/{repository}/{commit}/data/{name}",
                      raw_dir / name) for name in SOURCE_FILES]
    for name in ("sparseRRR.py", "rnaseqTools.py"):
        files.append(download(f"https://raw.githubusercontent.com/{repository}/{commit}/code/{name}",
                              root / "vendor" / "mini-atlas" / "code" / name))
    manifest = {"repository": repository, "commit": commit, "files": files}
    manifest_path = raw_dir / "download_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior = {Path(row["path"]).name: row["sha256"] for row in previous["files"]}
        for row in files:
            if prior.get(Path(row["path"]).name, row["sha256"]) != row["sha256"]:
                raise ValueError(f"Previously downloaded asset changed: {row['path']}")
    atomic_json(manifest_path, manifest)
    return manifest


def clone_vendor(url: str, destination: Path, revision: str | None = None) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        subprocess.run(["git", "clone", "--depth", "1", url, str(destination)], check=True)
    if not (destination / ".git").exists():
        raise ValueError(f"Vendor destination is not a checkout: {destination}")
    if revision:
        current = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        if current != revision:
            subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision], check=True)
            subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)
    commit = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    return {"url": url, "path": str(destination), "commit": commit}


def hf_snapshot(repo_id: str, destination: Path, cache: Path, revision: str | None = None) -> dict:
    from huggingface_hub import HfApi, snapshot_download
    commit = HfApi().model_info(repo_id, revision=revision).sha
    snapshot_download(repo_id, revision=commit, local_dir=str(destination), cache_dir=str(cache))
    files = [{"path": str(p.relative_to(destination)), "sha256": file_sha256(p)}
             for p in sorted(destination.rglob("*")) if p.is_file() and ".cache" not in p.parts]
    return {"repo_id": repo_id, "revision": commit, "path": str(destination), "files": files}


def verify_resource_record(record: dict, *, base: Path | None = None) -> None:
    """Verify recorded artifacts without redownloading or regenerating them."""
    if not isinstance(record, dict):
        raise ValueError("Invalid resource provenance record")
    pairs = (("path", "sha256"), ("vectors_path", "vectors_sha256"),
             ("table", "table_sha256"), ("gene_ids_path", "gene_ids_sha256"))
    for path_key, hash_key in pairs:
        if path_key in record and hash_key in record:
            path = Path(record[path_key])
            if base is not None and not path.is_absolute():
                path = base / path
            if not path.is_file() or file_sha256(path) != record[hash_key]:
                raise ValueError(f"Frozen resource is missing or changed: {path}; restore the recorded asset")
    if {"url", "path", "commit"} <= set(record):
        repo = Path(record["path"])
        if not (repo / ".git").exists():
            raise ValueError(f"Frozen vendor checkout is missing: {repo}")
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if actual != record["commit"]:
            raise ValueError(f"Frozen vendor revision changed: {repo}; restore commit {record['commit']}")
    # HF file records are relative to the snapshot; extracted UCE files use names.
    children_base = Path(record["path"]) if "repo_id" in record else base
    if "name" in record and "sha256" in record and "path" not in record:
        if base is None:
            raise ValueError("Extracted resource record lacks its base directory")
        path = base / record["name"]
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Frozen resource is missing or changed: {path}; restore the recorded asset")
    for key, value in record.items():
        if isinstance(value, dict):
            verify_resource_record(value, base=children_base)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, dict):
                    child_base = children_base
                    if key == "protein_files":
                        archives = [row for row in record["files"]
                                    if Path(row["path"]).name == "protein_embeddings.tar.gz"]
                        if len(archives) != 1:
                            raise ValueError("UCE provenance lacks the protein archive")
                        child_base = Path(archives[0]["path"]).parent / "protein_embeddings"
                    verify_resource_record(child, base=child_base)


def fetch_orthologs(destination: Path) -> dict:
    """Freeze a BioMart snapshot; only unambiguous one-to-one symbol pairs survive."""
    manifest_path = destination.with_suffix(".manifest.json")
    if manifest_path.is_file():
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        if Path(record["table"]).resolve() != destination.resolve():
            raise ValueError("Frozen ortholog snapshot does not match its configured destination")
        verify_resource_record(record)
        return record
    source = destination.with_suffix(".biomart.tsv")
    query = ('<?xml version="1.0" encoding="UTF-8"?>'
             '<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" '
             'count="" datasetConfigVersion="0.6"><Dataset name="mmusculus_gene_ensembl" interface="default">'
             '<Attribute name="external_gene_name"/>'
             '<Attribute name="hsapiens_homolog_associated_gene_name"/>'
             '<Attribute name="hsapiens_homolog_orthology_type"/>'
             '<Attribute name="ensembl_gene_id"/>'
             '<Attribute name="hsapiens_homolog_ensembl_gene"/></Dataset></Query>')
    record = download("https://www.ensembl.org/biomart/martservice?" + urllib.parse.urlencode({"query": query}), source)
    pairs = set()
    annotations = set()
    for row in csv.reader(io.StringIO(source.read_text(encoding="utf-8")), delimiter="\t"):
        if len(row) >= 5 and row[0] and row[3]:
            annotations.add((row[0], row[3], row[1], row[4], row[2]))
        if len(row) >= 3 and row[0] and row[1] and row[2] == "ortholog_one2one":
            pairs.add((row[0], row[1]))
    from collections import Counter
    mouse_counts = Counter(a for a, _ in pairs)
    human_counts = Counter(b for _, b in pairs)
    pairs = sorted((a, b) for a, b in pairs if mouse_counts[a] == human_counts[b] == 1)
    if not pairs:
        raise ValueError("BioMart returned no one-to-one orthologs; inspect the saved source response")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["mouse_symbol", "human_symbol"])
        writer.writerows(pairs)
    gene_ids = destination.parent / "mouse_gene_ids.tsv"
    with gene_ids.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["mouse_symbol", "mouse_ensembl", "human_symbol", "human_ensembl", "orthology_type"])
        writer.writerows(sorted(annotations))
    result = {**record, "pairs": len(pairs), "table": str(destination),
              "table_sha256": file_sha256(destination), "gene_ids_path": str(gene_ids),
              "gene_ids_sha256": file_sha256(gene_ids)}
    atomic_json(manifest_path, result)
    return result


_NICHEFORMER_MEAN_COMMIT = "485cadbc5caa15119adfd54228f8a8af835fcabc"
_NICHEFORMER_MEAN_BLOB = "ed8548c5ac014a676a639f3214a419c753e5f323"
_NICHEFORMER_MEAN_POLICY = "dissociated_notebook_half_up_min_one"


def _prepare_nicheformer_mean(raw: Path, destination: Path, reference_genes: int) -> dict:
    """Apply the authors' dissociated notebook policy without modifying raw bytes."""
    import numpy as np

    values = np.asarray(np.load(raw, allow_pickle=False), dtype=np.float64)
    if values.ndim != 1 or values.shape != (reference_genes,):
        raise ValueError("Nicheformer raw mean must match the native reference gene count exactly")
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("Nicheformer raw means contain infinity or negative values")
    # The official dissociated mouse-brain notebook sanitizes, rounds half-up,
    # and substitutes one for rounded zeros before passing means to the tokenizer.
    clean = np.nan_to_num(values, nan=0.0)
    rounded = np.where((clean % 1) >= 0.5, np.ceil(clean), np.floor(clean))
    prepared = np.where(rounded == 0, 1.0, rounded)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".prepared.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, prepared, allow_pickle=False)
        raw_sha = file_sha256(raw)
        prepared_sha = file_sha256(temporary)
        if destination.exists():
            current = file_sha256(destination)
            if current not in {raw_sha, prepared_sha}:
                raise ValueError("Existing Nicheformer means differ from both the raw source and prepared means; "
                                 "preserve the existing file and inspect its provenance")
            if current == prepared_sha:
                temporary.unlink()
            else:
                os.replace(temporary, destination)
        else:
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"policy": _NICHEFORMER_MEAN_POLICY, "reference_genes": reference_genes,
            "raw_nan_count": int(np.isnan(values).sum()),
            "rounded_zero_count": int((rounded == 0).sum()),
            "source": f"https://github.com/theislab/nicheformer/blob/{_NICHEFORMER_MEAN_COMMIT}/"
                      "notebooks/tokenization/scRNAseq_mouse_brain_haviv.ipynb#L502-L504",
            "raw_sha256": raw_sha, "prepared_sha256": prepared_sha}


def prefetch_nicheformer_resources(config: dict, *, root: Path) -> dict:
    """Freeze original means and the explicit author-normalized denominator array.

    Keep this preparation outside embeddings.py so frozen GenePT/UCE encoder
    fingerprints remain unchanged. Existing complete resource manifests are verified.
    """
    import anndata as ad
    from .embeddings import build_nicheformer_gene_mapping

    destination = resolve(root, config["technology_mean_path"])
    manifest_path = destination.parent / "prefetch.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("normalization", {}).get("policy") != _NICHEFORMER_MEAN_POLICY:
            raise ValueError("Existing frozen Nicheformer resources use another mean policy; inspect their provenance")
        if Path(previous["technology_mean"]["path"]).resolve() != destination.resolve():
            raise ValueError("Frozen Nicheformer means do not match the configured path")
        verify_resource_record(previous)
        return previous
    repo = resolve(root, config.get("repo_dir", "vendor/Nicheformer"))
    vendor = clone_vendor("https://github.com/theislab/nicheformer.git", repo,
                          config.get("vendor_revision"))
    raw = destination.with_name(destination.stem + ".raw.npy")
    raw_record = download(f"https://raw.githubusercontent.com/theislab/nicheformer/{_NICHEFORMER_MEAN_COMMIT}/"
                          "data/model_means/dissociated_mean_script.npy", raw)
    payload = raw.read_bytes()
    blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    if blob != _NICHEFORMER_MEAN_BLOB:
        raise ValueError("Nicheformer raw mean differs from the pinned official Git blob; preserve it for inspection")
    reference = ad.read_h5ad(resolve(root, config["checkpoint_dir"]) / "model.h5ad", backed="r")
    try:
        reference_genes = reference.n_vars
    finally:
        if getattr(reference, "file", None) is not None:
            reference.file.close()
    normalization = _prepare_nicheformer_mean(raw, destination, reference_genes)
    mapping = build_nicheformer_gene_mapping(config, root=root)
    result = {"vendor": vendor, "raw_technology_mean": raw_record,
              "technology_mean": {"path": str(destination), "sha256": file_sha256(destination),
                                  "bytes": destination.stat().st_size},
              "normalization": normalization, "gene_mapping": mapping}
    atomic_json(manifest_path, result)
    return result


def prefetch_models(config: dict, root: Path, names: list[str]) -> dict:
    """Download authentic checkpoints; store immutable revisions for subsequent offline jobs."""
    cache = resolve(root, config["paths"]["cache_dir"])
    os.environ.pop("HF_HUB_OFFLINE", None)
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    vendors = {
        "limix": ("https://github.com/limix-ldm-ai/LimiX.git", config["models"]["limix"]["repo_dir"]),
        "uce": ("https://github.com/snap-stanford/UCE.git", config["embeddings"]["uce"]["repo_dir"]),
    }
    hf_models = {
        "tabpfn": ("Prior-Labs/tabpfn_3_5", Path(config["models"]["tabpfn"]["checkpoint"]).parent),
        "limix": ("stable-ai/LimiX-2", Path(config["models"]["limix"]["checkpoint"]).parent),
        "kumo": ("nvidia/Kumo-Tabular", Path(config["models"]["kumo"]["checkpoint_dir"])),
        "nicheformer": ("theislab/Nicheformer", Path(config["embeddings"]["nicheformer"]["checkpoint_dir"])),
    }
    required_keys = {"tabpfn": {"checkpoint"}, "limix": {"vendor", "checkpoint"},
                     "kumo": {"checkpoint"}, "genept": {"vectors", "orthologs"},
                     "uce": {"vendor", "resources"}, "nicheformer": {"checkpoint", "resources"}}
    unknown = set(names) - set(required_keys)
    if unknown:
        raise ValueError(f"Unknown model resource names: {sorted(unknown)}")
    reusable = set()
    for name in names:
        record = manifest.get(name)
        if record is None:
            continue
        if not required_keys[name] <= set(record):
            raise ValueError(f"Incomplete frozen {name} manifest entry; inspect and restore its provenance before setup")
        if name in hf_models:
            repo_id, relative = hf_models[name]
            checkpoint = record["checkpoint"]
            if (checkpoint.get("repo_id") != repo_id or
                    Path(checkpoint["path"]).resolve() != resolve(root, relative).resolve()):
                raise ValueError(f"Frozen {name} checkpoint does not match the configured repository/path")
        if name in vendors and Path(record["vendor"]["path"]).resolve() != resolve(root, vendors[name][1]).resolve():
            raise ValueError(f"Frozen {name} vendor does not match its configured path")
        if name == "genept":
            for recorded, configured in ((record["vectors"]["vectors_path"], config["embeddings"][name]["vectors_path"]),
                                         (record["orthologs"]["table"], config["embeddings"][name]["orthologs_path"])):
                if Path(recorded).resolve() != resolve(root, configured).resolve():
                    raise ValueError("Frozen GenePT resources do not match the configured paths")
        verify_resource_record(record)
        reusable.add(name)
    ortholog_record = None
    if (set(names) - reusable) & {"genept", "nicheformer"}:
        ortholog_record = fetch_orthologs(resolve(root, config["embeddings"]["genept"]["orthologs_path"]))
    for name in names:
        if name in reusable:
            continue
        record = dict(manifest.get(name, {}))
        if name in vendors:
            url, relative = vendors[name]
            record["vendor"] = clone_vendor(url, resolve(root, relative), record.get("vendor", {}).get("commit"))
        if name in hf_models:
            repo_id, relative = hf_models[name]
            record["checkpoint"] = hf_snapshot(repo_id, resolve(root, relative), cache / "huggingface" / "hub",
                                               "v1.0.1" if name == "kumo" else None)
        if name == "genept":
            from .embeddings import prefetch_genept
            record["vectors"] = prefetch_genept(config["embeddings"][name], root=root)
            record["orthologs"] = ortholog_record
        elif name == "uce":
            from .embeddings import prefetch_uce
            record["resources"] = prefetch_uce(config["embeddings"][name], root=root)
        elif name == "nicheformer":
            record["resources"] = prefetch_nicheformer_resources(config["embeddings"][name], root=root)
        elif name not in hf_models:
            raise ValueError(f"Unknown model resource name: {name}")
        manifest[name] = record
        atomic_json(manifest_path, manifest)
    return manifest
