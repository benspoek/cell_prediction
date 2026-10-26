"""Explicit, resumable downloads. Run this module on the cluster, never inside a fit."""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from .config import atomic_json, file_sha256, resolve

SOURCE_FILES = (
    "m1_patchseq_exon_counts.csv.gz", "m1_patchseq_intron_counts.csv.gz",
    "m1_patchseq_meta_data.csv", "m1_patchseq_ephys_features.csv",
    "gene_lengths.txt", "ion-channel-genes-group-177.csv",
)


def download(url: str, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        temporary = destination.with_name(destination.name + ".download")
        request = urllib.request.Request(url, headers={"User-Agent": "cell-property-prediction/0.1"})
        with urllib.request.urlopen(request, timeout=180) as source, temporary.open("wb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
        os.replace(temporary, destination)
    return {"url": url, "path": str(destination), "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size}


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
            from .embeddings import prefetch_nicheformer
            record["resources"] = prefetch_nicheformer(config["embeddings"][name], root=root)
        elif name not in hf_models:
            raise ValueError(f"Unknown model resource name: {name}")
        manifest[name] = record
        atomic_json(manifest_path, manifest)
    return manifest
