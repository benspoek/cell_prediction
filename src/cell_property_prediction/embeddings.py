"""Frozen cell embeddings with explicit local resources and cell-ID preservation.

Embedding jobs never receive electrophysiology, cell-type, donor or fold labels.
Only the explicit prefetch functions may access the network.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from scipy import sparse

from .config import atomic_json, file_sha256, resolve
from .models import _git_revision, _offline_environment, _resource, _version

GENEPT_ARCHIVE_MD5 = "3f6ce4317e3a0091978ae5cb8fbf05a3"
GENEPT_VECTOR_FILENAME = "GenePT_gene_protein_embedding_model_3_text.pickle"


@dataclass(frozen=True)
class EmbeddingResult:
    path: Path
    audit: dict


def _save(name: str, values, cell_ids, output_dir: Path, audit: dict) -> EmbeddingResult:
    values = np.asarray(values, dtype=np.float32)
    ids = np.asarray(cell_ids, dtype=str)
    if values.ndim != 2 or values.shape[0] != len(ids) or not np.isfinite(values).all():
        raise ValueError("Embedding must be a finite cell-by-feature matrix")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Embedding cell IDs must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.npz"
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, values=values, cell_ids=ids,
                            feature_names=np.asarray([f"{name}_{i:04d}" for i in range(values.shape[1])]))
    os.replace(temporary, path)
    audit.update(rows=len(ids), dimensions=values.shape[1], output_sha256=file_sha256(path),
                 labels_used=False, frozen=True)
    atomic_json(output_dir / f"{name}.audit.json", audit)
    return EmbeddingResult(path, audit)


def _load_prepared(path: Path):
    import anndata as ad

    data = ad.read_h5ad(path)
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Prepared cell and gene names must be unique")
    if set(data.obs.columns) - {"species"}:
        raise ValueError("Embedding input may contain only the species obs column; strip all outcome/identity labels")
    if "species" in data.obs and not data.obs["species"].astype(str).eq("mouse").all():
        raise ValueError("This benchmark requires mouse cells")
    counts = sparse.csr_matrix(data.X, dtype=np.float64)
    counts.eliminate_zeros()
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("Prepared X must contain finite, nonnegative raw counts")
    if not np.allclose(counts.data, np.round(counts.data), atol=1e-6, rtol=0):
        raise ValueError("Prepared X must be unnormalised integer-valued counts")
    if (np.asarray(counts.sum(axis=1)).ravel() <= 0).any():
        raise ValueError("Zero-count cells cannot be embedded")
    data.X = counts
    return data


def load_genept_vectors(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an upstream trusted dictionary pickle or the safe converted NPZ."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if {"gene_symbols", "vectors"} <= set(archive.files):
                names, vectors = archive["gene_symbols"], archive["vectors"]
            elif {"genes", "embeddings"} <= set(archive.files):
                names, vectors = archive["genes"], archive["embeddings"]
            else:
                raise ValueError("GenePT NPZ requires gene_symbols/vectors (or genes/embeddings)")
    else:
        # Only use the explicitly downloaded, checksummed upstream artifact here.
        with path.open("rb") as handle:
            source = pickle.load(handle)
        if not isinstance(source, dict) or not source:
            raise ValueError("Upstream GenePT pickle must be a nonempty gene->vector dictionary")
        names = np.asarray(sorted(source), dtype=str)
        vectors = np.stack([np.asarray(source[name]).reshape(-1) for name in names])
    names, vectors = np.asarray(names, dtype=str), np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(names), 3072) or not np.isfinite(vectors).all():
        raise ValueError("Expected the GenePT v2 text-embedding-3-large 3072-dimensional release")
    if len(set(names)) != len(names):
        raise ValueError("GenePT vectors contain duplicate gene symbols")
    return names, vectors


def _genept(data, config, root):
    vectors_path = _resource(root, config["vectors_path"])
    ortholog_path = _resource(root, config["orthologs_path"])
    names, vectors = load_genept_vectors(vectors_path)
    orthologs = pd.read_csv(ortholog_path, sep="\t", dtype=str).fillna("")
    required = {"mouse_symbol", "human_symbol"}
    if not required <= set(orthologs) or orthologs[list(required)].eq("").any().any():
        raise ValueError("Ortholog TSV needs nonempty mouse_symbol and human_symbol columns")
    if orthologs.mouse_symbol.duplicated().any() or orthologs.human_symbol.duplicated().any():
        raise ValueError("The ortholog table must be unambiguous one-to-one before encoding")
    mapping = dict(zip(orthologs.mouse_symbol, orthologs.human_symbol))
    lookup = {name: index for index, name in enumerate(names)}
    pairs = [(i, lookup[mapping[gene]]) for i, gene in enumerate(data.var_names)
             if gene in mapping and mapping[gene] in lookup]
    if not pairs:
        raise ValueError("No prepared mouse genes matched the fixed human GenePT vectors")
    columns, embedding_rows = map(np.asarray, zip(*pairs))
    totals = np.asarray(data.X.sum(axis=1)).ravel()
    total = float(config.get("normalisation_total", 10000))
    expression = sparse.diags(total / totals) @ data.X
    expression.data = np.log1p(expression.data)
    matched = expression[:, columns]
    numerator = np.asarray(matched @ vectors[embedding_rows], dtype=np.float32)
    policy = config.get("pooling", "gene_count_mean")
    if policy == "gene_count_mean":
        # Original GenePT notebook: all input genes, including zero-vector missing genes.
        values = numerator / data.n_vars
        denominator = f"all {data.n_vars} input genes; unmatched vectors are zero"
    elif policy == "expression_weighted_mean":
        weights = np.asarray(matched.sum(axis=1)).ravel()
        if (weights <= 0).any():
            raise ValueError("At least one cell has no matched expressed GenePT genes")
        values = numerator / weights[:, None]
        denominator = "sum of matched log-normalised expression per cell"
    else:
        raise ValueError(f"Unknown GenePT pooling policy {policy!r}")
    raw_matched = np.asarray(data.X[:, columns].sum(axis=1)).ravel() / totals
    audit = {"vectors_path": str(vectors_path), "vectors_sha256": file_sha256(vectors_path),
             "orthologs_path": str(ortholog_path), "orthologs_sha256": file_sha256(ortholog_path),
             "pooling": policy, "denominator": denominator, "normalisation_total": total,
             "normalisation": "full-input cell total, then log1p; no cohort fitting",
             "matched_genes": len(pairs), "input_genes": data.n_vars,
             "matched_mouse_genes": data.var_names[columns].tolist(),
             "retained_raw_count_fraction_min": float(raw_matched.min()),
             "retained_raw_count_fraction_median": float(np.median(raw_matched))}
    return values, audit


def _nicheformer(data, config, root, device, offline):
    import anndata as ad
    import torch
    from transformers import AutoModelForMaskedLM

    checkpoint_dir = _resource(root, config["checkpoint_dir"], directory=True)
    mean_path = _resource(root, config["technology_mean_path"])
    required = ("config.json", "model.safetensors", "model.h5ad", "vocab.json",
                "configuration_nicheformer.py", "modeling_nicheformer.py")
    files = {name: _resource(checkpoint_dir, name) for name in required}
    reference = ad.read_h5ad(files["model.h5ad"])
    vocabulary = json.loads(files["vocab.json"].read_text(encoding="utf-8"))
    genes = reference.var_names.astype(str).tolist()
    if not genes or any(vocabulary.get(gene) != i + 30 for i, gene in enumerate(genes)):
        raise ValueError("Nicheformer reference gene order and native token IDs disagree")
    means = np.asarray(np.load(mean_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    if means.shape != (len(genes),) or not np.isfinite(means).all() or (means < 0).any():
        raise ValueError("Dissociated technology mean must align exactly with the native reference gene order")
    gene_index = {gene: i for i, gene in enumerate(genes)}
    source_map = {}
    map_path = config.get("gene_mapping_path")
    map_audit = None
    if map_path:
        path = _resource(root, map_path)
        frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        if "mouse_symbol" not in frame:
            raise ValueError("Nicheformer mapping TSV requires mouse_symbol")
        if frame.mouse_symbol.duplicated().any():
            raise ValueError("Nicheformer mapping must be unique per mouse symbol")
        for row in frame.to_dict("records"):
            candidates = [row.get(field, "") for field in ("model_gene_id", "mouse_ensembl", "human_ensembl")]
            matched = next((gene for gene in candidates if gene in gene_index), None)
            if matched is not None:
                source_map[row["mouse_symbol"]] = matched
        map_audit = {"path": str(path), "sha256": file_sha256(path)}
    else:
        # Native references can carry symbol aliases; never infer symbols by case.
        for column in ("mouse_symbol", "gene_symbols", "gene_symbol", "feature_name", "gene_name"):
            if column in reference.var:
                aliases = reference.var[column].astype(str)
                unique = ~aliases.duplicated(keep=False)
                source_map.update({alias: gene for alias, gene, valid in zip(aliases, genes, unique)
                                   if valid and alias and alias != "nan"})
    pairs = [(i, gene_index[source_map[gene]]) for i, gene in enumerate(data.var_names) if gene in source_map]
    if not pairs:
        raise ValueError("No Nicheformer gene mapping; prefetch a mouse_symbol/model_gene_id TSV")
    input_indices, reference_indices = map(np.asarray, zip(*pairs))
    if len(np.unique(reference_indices)) != len(reference_indices):
        raise ValueError("Multiple prepared genes map to one native Nicheformer token; resolve mapping explicitly")
    aligned = sparse.csr_matrix((data.n_obs, len(genes)), dtype=np.float64).tolil()
    aligned[:, reference_indices] = data.X[:, input_indices]
    aligned = aligned.tocsr()
    counts = np.asarray(aligned.sum(axis=1)).ravel()
    if (counts <= 0).any():
        raise ValueError("A cell has no expression covered by Nicheformer")
    normalised = sparse.diags(10000 / counts) @ aligned
    normalised = normalised.multiply(1 / np.maximum(means, 1e-6)).tocsr()
    tokens = np.zeros((data.n_obs, 1500), dtype=np.int64)
    assay = int(config.get("assay_token", 18))
    if assay not in range(7, 19):
        raise ValueError("Invalid native Nicheformer assay token")
    tokens[:, :3] = [6, assay, 3]  # official HF implementation: species, assay, modality
    expressed = []
    for i in range(data.n_obs):
        row = normalised.getrow(i)
        order = np.argsort(-row.data, kind="stable")[:1497]
        ranked = row.indices[order] + 30
        tokens[i, 3:3 + len(ranked)] = ranked
        expressed.append(len(ranked))
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    model, loading = AutoModelForMaskedLM.from_pretrained(
        str(checkpoint_dir), trust_remote_code=True, local_files_only=offline, output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Nicheformer checkpoint did not load exactly: {loading}")
    model.to(device).eval()
    batch_size = int(config.get("batch_size", 16))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches = []
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            batch = torch.as_tensor(tokens[start:start + batch_size], device=device)
            # Use the native HF helper verbatim, including its pooling over padded positions.
            embeddings = model.get_embeddings(input_ids=batch, attention_mask=batch.ne(0),
                                               layer=-1, with_context=False)
            batches.append(embeddings.detach().float().cpu().numpy())
    values = np.vstack(batches)
    if values.shape[1] != 512:
        raise ValueError("Nicheformer must yield native 512-dimensional representations")
    return values, {"checkpoint_files": {name: file_sha256(path) for name, path in files.items()},
                    "technology_mean_sha256": file_sha256(mean_path), "gene_mapping": map_audit,
                    "matched_genes": len(pairs), "input_genes": data.n_vars,
                    "token_order": ["mouse=6", f"assay={assay}", "dissociated=3"],
                    "assay_mapping": "available pretrained Smart-seq token 18" if assay == 18 else str(assay),
                    "context_length": 1500, "gene_tokens_max": 1497, "seed": seed,
                    "pooling": "official HF get_embeddings(layer=-1,with_context=False), native unmasked mean",
                    "transformers_version": _version("transformers"), "torch_version": _version("torch")}


def _uce(data, config, root, device, offline, work_dir):
    import anndata as ad

    repo = _resource(root, config["repo_dir"], directory=True)
    checkpoint = _resource(root, config["checkpoint"])
    resources = _resource(root, config["resources_dir"], directory=True)
    script = _resource(repo, "eval_single_anndata.py")
    paths = {name: _resource(resources, name) for name in
             ("species_chrom.csv", "species_offsets.pkl", "all_tokens.torch")}
    protein_dir = _resource(resources, "protein_embeddings", directory=True)
    if not list(protein_dir.glob("*.pt")):
        raise FileNotFoundError("UCE protein embedding archive has not been extracted")
    work_dir.mkdir(parents=True, exist_ok=True)
    clean = ad.AnnData(X=data.X.copy(), obs=pd.DataFrame(index=data.obs_names),
                       var=pd.DataFrame(index=data.var_names))
    input_path = work_dir / "counts.h5ad"
    clean.write_h5ad(input_path)
    arguments = {"adata_path": input_path.as_posix(), "dir": work_dir.as_posix() + "/", "species": "mouse",
                 "filter": False, "skip": False, "model_loc": checkpoint.as_posix(),
                 "batch_size": int(config.get("batch_size", 16)), "pad_length": 1536,
                 "pad_token_idx": 0, "chrom_token_left_idx": 1, "chrom_token_right_idx": 2,
                 "cls_token_idx": 3, "CHROM_TOKEN_OFFSET": 143574, "sample_size": 1024,
                 "CXG": True, "nlayers": 33, "output_dim": 1280, "d_hid": 5120,
                 "token_dim": 5120, "multi_gpu": False,
                 "spec_chrom_csv_path": paths["species_chrom.csv"].as_posix(),
                 "token_file": paths["all_tokens.torch"].as_posix(),
                 "protein_embeddings_dir": protein_dir.as_posix(),
                 "offset_pkl_path": paths["species_offsets.pkl"].as_posix()}
    if not str(device).startswith(("cpu", "cuda")):
        raise ValueError("UCE supports CPU or CUDA")
    request_path = work_dir / "request.json"
    atomic_json(request_path, {"arguments": arguments, "seed": int(config.get("seed", 42)),
                               "offline": offline, "cpu": str(device) == "cpu"})
    # Run the authentic vendor entry point in an isolated process. Resource-path
    # rebasing and a no-download check are the only modifications to vendor code.
    runner = r'''
import importlib.util, json, pathlib, random, sys
from types import SimpleNamespace
import numpy as np
import torch
from accelerate import Accelerator
request=json.load(open(sys.argv[1],encoding="utf-8"))
args=SimpleNamespace(**request["arguments"])
random.seed(request["seed"]); np.random.seed(request["seed"]); torch.manual_seed(request["seed"])
sys.path.insert(0,str(pathlib.Path.cwd()))
import evaluate
import data_proc.gene_embeddings as ge
ge.MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH["ESM2"] = {
    key:pathlib.Path(args.protein_embeddings_dir)/path.name
    for key,path in ge.MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH["ESM2"].items()}
if request["offline"]:
    def existing_only(url,path):
        if not pathlib.Path(path).is_file(): raise FileNotFoundError("Offline UCE resource missing: "+str(path))
    evaluate.figshare_download=existing_only
spec=importlib.util.spec_from_file_location("uce_official_entry",sys.argv[2])
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.main(args,Accelerator(project_dir=args.dir,cpu=request["cpu"]))
'''
    env = os.environ.copy()
    if str(device) == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif ":" in str(device):
        # Preserve Slurm's physical GPU allocation; select a logical device only.
        env["ACCELERATE_TORCH_DEVICE"] = str(device)
    log_path = work_dir / "vendor.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run([str(config.get("python", sys.executable)), "-c", runner,
                        str(request_path), str(script)], cwd=repo, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    output = _resource(work_dir, "counts_uce_adata.h5ad")
    encoded = ad.read_h5ad(output)
    if set(encoded.obs_names) != set(data.obs_names):
        raise ValueError("UCE dropped or added cells; refuse to align labels by position")
    encoded = encoded[data.obs_names]
    values = np.asarray(encoded.obsm["X_uce"], dtype=np.float32)
    if values.shape != (data.n_obs, 1280):
        raise ValueError("Expected the native UCE33 1280-dimensional output")
    return values, {"checkpoint_sha256": file_sha256(checkpoint), "vendor_commit": _git_revision(repo),
                    "script_sha256": file_sha256(script), "resource_hashes":
                    {name: file_sha256(path) for name, path in paths.items()},
                    "native_layers": 33, "sample_size": 1024, "seed": config.get("seed", 42),
                    "additional_filtering": False, "matched_genes": encoded.n_vars,
                    "vendor_log": str(log_path), "vendor_arguments": arguments}


def _embedding_context(name: str, config: dict, root: Path, path: Path, device: str) -> dict:
    """Bind cached output to the exact input, implementation and local resources."""
    keys = {"genept": ("vectors_path", "orthologs_path"),
            "uce": ("checkpoint",),
            "nicheformer": ("technology_mean_path", "gene_mapping_path")}
    if name not in keys:
        raise ValueError(f"Unknown embedding {name!r}")
    files = [_resource(root, config[key]) for key in keys[name]]
    if name == "nicheformer":
        checkpoint = _resource(root, config["checkpoint_dir"], directory=True)
        files.extend(_resource(checkpoint, value) for value in
                     ("config.json", "model.safetensors", "model.h5ad", "vocab.json",
                      "configuration_nicheformer.py", "modeling_nicheformer.py"))
    elif name == "uce":
        repo = _resource(root, config["repo_dir"], directory=True)
        files.extend(sorted(repo.rglob("*.py")))
        resources = _resource(root, config["resources_dir"], directory=True)
        files.extend(_resource(resources, value) for value in
                     ("species_chrom.csv", "species_offsets.pkl", "all_tokens.torch"))
        files.extend(sorted((resources / "protein_embeddings").glob("*.pt")))
    return {"schema_version": 1, "embedding": name, "config": config,
            "prepared_sha256": file_sha256(path), "implementation_sha256": file_sha256(Path(__file__)),
            "device": str(device), "python": sys.version,
            "packages": {package: _version(package) for package in
                         ("numpy", "pandas", "scipy", "anndata", "torch", "transformers", "accelerate")},
            "resources": {str(item): file_sha256(item) for item in sorted(set(files))}}


def generate_embedding(name: str, config: dict, prepared_h5ad_path: str | Path, output_dir: str | Path,
                       *, device: str, offline: bool = True, root: Path = Path(".")) -> EmbeddingResult:
    root = Path(root).resolve()
    path = _resource(root, prepared_h5ad_path)
    output_dir = resolve(root, output_dir).resolve()
    _offline_environment(root / "cache", offline)
    started = time.perf_counter()
    data = _load_prepared(path)
    context = _embedding_context(name, config, root, path, device)
    context_hash = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    dimensions = {"genept": 3072, "uce": 1280, "nicheformer": 512}[name]
    output = output_dir / f"{name}.npz"
    audit_path = output_dir / f"{name}.audit.json"
    if output.exists() or audit_path.exists():
        if not (output.is_file() and audit_path.is_file()):
            raise ValueError(f"Incomplete frozen {name} embedding; inspect/archive both output and audit before regenerating")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("context_sha256") != context_hash:
            raise ValueError(f"Frozen {name} embedding context changed; use a separate output directory or archive it explicitly")
        if audit.get("output_sha256") != file_sha256(output):
            raise ValueError(f"Frozen {name} embedding checksum changed")
        with np.load(output, allow_pickle=False) as archive:
            values, ids = archive["values"], archive["cell_ids"]
            expected_ids = np.asarray(data.obs_names, dtype=str)
            if (values.shape != (data.n_obs, dimensions) or not np.isfinite(values).all() or
                    not np.array_equal(ids, expected_ids) or len(np.unique(ids)) != len(ids) or
                    audit.get("dimensions") != dimensions or audit.get("rows") != data.n_obs):
                raise ValueError(f"Frozen {name} embedding dimensions or cell IDs do not match its prepared input")
        return EmbeddingResult(output, audit)
    if name == "genept":
        values, audit = _genept(data, config, root)
    elif name == "nicheformer":
        values, audit = _nicheformer(data, config, root, device, offline)
    elif name == "uce":
        values, audit = _uce(data, config, root, device, offline, output_dir / "uce_work")
    else:
        raise ValueError(f"Unknown embedding {name!r}")
    if np.asarray(values).shape != (data.n_obs, dimensions):
        raise ValueError(f"Expected native {name} output shape {(data.n_obs, dimensions)}")
    audit.update(embedding=name, prepared_sha256=file_sha256(path), device=str(device), offline=offline,
                 elapsed_seconds=time.perf_counter() - started, numpy_version=_version("numpy"),
                 context=context, context_sha256=context_hash)
    return _save(name, values, data.obs_names, output_dir, audit)


def prefetch_genept(config: dict, *, root: Path) -> dict:
    """Download the fixed GenePT v2 archive, validate its published MD5, convert safely."""
    from .assets import download

    destination = resolve(root, config["vectors_path"])
    archive = destination.parent / "GenePT_emebdding_v2.zip"
    record = download("https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1", archive)
    digest = hashlib.md5()  # Upstream publishes MD5; SHA256 is recorded separately.
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != GENEPT_ARCHIVE_MD5:
        raise ValueError("GenePT archive differs from the published v2 MD5")
    name = GENEPT_VECTOR_FILENAME
    with zipfile.ZipFile(archive) as handle:
        # The published v2 ZIP has a trailing dot after .pickle, although the
        # Zenodo description omits it. Write to our canonical local name.
        members = [member for member in handle.infolist() if not member.is_dir()
                   and "__MACOSX" not in PurePosixPath(member.filename).parts
                   and not PurePosixPath(member.filename).name.startswith("._")]
        candidates = [member for member in members
                      if PurePosixPath(member.filename).name in {name, name + "."}]
        if len(candidates) != 1:
            observed = [member.filename for member in members if ".pickle" in member.filename]
            raise ValueError(f"GenePT archive must contain one exact model-3 vector pickle "
                             f"({name} or {name}.); found {len(candidates)} matches. "
                             f"Pickle members: {observed}")
        member = candidates[0]
        raw = destination.parent / name
        with handle.open(member) as source, raw.open("wb") as output:
            import shutil
            shutil.copyfileobj(source, output)
    names, vectors = load_genept_vectors(raw)
    if destination.suffix == ".npz":
        with destination.open("wb") as handle:
            np.savez_compressed(handle, gene_symbols=names, vectors=vectors)
    elif destination.resolve() != raw.resolve():
        import shutil
        shutil.copyfile(raw, destination)
    return {"archive": record, "archive_member": member.filename, "published_md5": digest.hexdigest(),
            "genes": len(names), "dimensions": 3072,
            "vectors_path": str(destination), "vectors_sha256": file_sha256(destination)}


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    import shutil

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        if not members:
            raise ValueError("Empty UCE protein archive")
        if any(not Path(member.name).parts for member in members):
            raise ValueError("Empty archive member name")
        prefixed = all(Path(member.name).parts[0] == "protein_embeddings" for member in members)
        base = destination if prefixed else destination / "protein_embeddings"
        for member in members:
            target = (base / member.name).resolve()
            if not target.is_relative_to(destination) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise ValueError(f"Unreadable archive member {member.name!r}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
            else:
                raise ValueError(f"Unsupported archive member {member.name!r}")


def prefetch_uce(config: dict, *, root: Path) -> dict:
    from .assets import download

    resources = resolve(root, config["resources_dir"])
    checkpoint = resolve(root, config["checkpoint"])
    files = {"species_chrom.csv": "42706558", "species_offsets.pkl": "42706555",
             "all_tokens.torch": "42706585", "protein_embeddings.tar.gz": "42715213"}
    records = [download(f"https://figshare.com/ndownloader/files/{file_id}", resources / name)
               for name, file_id in files.items()]
    records.append(download("https://figshare.com/ndownloader/files/43423236", checkpoint))
    _safe_extract_tar(resources / "protein_embeddings.tar.gz", resources)
    protein_files = sorted((resources / "protein_embeddings").glob("*.pt"))
    if len(protein_files) < 8:
        raise ValueError("Expected UCE protein resources for all eight native species")
    return {"files": records, "protein_files": [{"name": path.name, "sha256": file_sha256(path)}
                                                   for path in protein_files], "layers": 33}


def prefetch_nicheformer(config: dict, *, root: Path) -> dict:
    """Freeze official technology means at the checked-out vendor revision."""
    from .assets import clone_vendor, download

    repo = resolve(root, config.get("repo_dir", "vendor/Nicheformer"))
    manifest_path = resolve(root, config["technology_mean_path"]).parent / "prefetch.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    record = clone_vendor("https://github.com/theislab/nicheformer.git", repo,
                          config.get("vendor_revision", previous.get("vendor", {}).get("commit")))
    mean = download(f"https://raw.githubusercontent.com/theislab/nicheformer/{record['commit']}/"
                    "data/model_means/dissociated_mean_script.npy",
                    resolve(root, config["technology_mean_path"]))
    means = np.load(resolve(root, config["technology_mean_path"]), allow_pickle=False)
    if means.ndim not in (1, 2) or not np.isfinite(means).all() or (means < 0).any():
        raise ValueError("Downloaded Nicheformer means are not a finite nonnegative array")
    mapping = build_nicheformer_gene_mapping(config, root=root)
    result = {"vendor": record, "technology_mean": mean, "gene_mapping": mapping}
    atomic_json(manifest_path, result)
    return result


def build_nicheformer_gene_mapping(config: dict, *, root: Path) -> dict:
    """Freeze symbols to actual native tokens using the downloaded BioMart snapshot.

    Native mouse IDs take precedence. Human IDs are accepted only for an explicit
    one-to-one ortholog. Ambiguous source symbols or target tokens are removed,
    rather than silently summing multiple input genes into a token.
    """
    import anndata as ad

    reference_path = _resource(root, Path(config["checkpoint_dir"]) / "model.h5ad")
    vocabulary_path = _resource(root, Path(config["checkpoint_dir"]) / "vocab.json")
    ids_path = _resource(root, config["gene_ids_path"])
    reference = ad.read_h5ad(reference_path, backed="r")
    genes = reference.var_names.astype(str).tolist()
    if getattr(reference, "file", None) is not None:
        reference.file.close()
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    native = set(genes) & set(vocabulary)
    if len(native) != len(genes):
        raise ValueError("Nicheformer reference IDs are not all present in its vocabulary")
    # Version suffix removal is explicit and does not merge reference tokens.
    normalized = {}
    for gene in native:
        stem = gene.split(".", 1)[0]
        if stem in normalized and normalized[stem] != gene:
            raise ValueError("Nicheformer reference has ambiguous version-stripped IDs")
        normalized[stem] = gene
    frame = pd.read_csv(ids_path, sep="\t", dtype=str).fillna("")
    required = {"mouse_symbol", "mouse_ensembl", "human_ensembl", "orthology_type"}
    if not required <= set(frame):
        raise ValueError(f"Gene-ID snapshot lacks columns: {sorted(required - set(frame))}")
    rows = []
    for row in frame.to_dict("records"):
        symbol = row["mouse_symbol"]
        if not symbol:
            continue
        mouse = normalized.get(row["mouse_ensembl"].split(".", 1)[0])
        human = normalized.get(row["human_ensembl"].split(".", 1)[0])
        if mouse is not None:
            rows.append((symbol, mouse, "native_mouse"))
        elif human is not None and row["orthology_type"] == "ortholog_one2one":
            rows.append((symbol, human, "one_to_one_human"))
    mapped = pd.DataFrame(rows, columns=["mouse_symbol", "model_gene_id", "mapping_kind"]).drop_duplicates()
    if mapped.empty:
        raise ValueError("No BioMart gene IDs match Nicheformer's native vocabulary")
    # A duplicate symbol's direct native entry has priority over ortholog entries.
    native_symbols = set(mapped.loc[mapped.mapping_kind.eq("native_mouse"), "mouse_symbol"])
    mapped = mapped.loc[mapped.mapping_kind.eq("native_mouse") | ~mapped.mouse_symbol.isin(native_symbols)]
    source_ambiguous = mapped.groupby("mouse_symbol").model_gene_id.nunique().gt(1)
    bad_sources = set(source_ambiguous[source_ambiguous].index)
    mapped = mapped.loc[~mapped.mouse_symbol.isin(bad_sources)].drop_duplicates(["mouse_symbol", "model_gene_id"])
    target_ambiguous = mapped.groupby("model_gene_id").mouse_symbol.nunique().gt(1)
    bad_targets = set(target_ambiguous[target_ambiguous].index)
    mapped = mapped.loc[~mapped.model_gene_id.isin(bad_targets)].sort_values("mouse_symbol")
    if mapped.empty:
        raise ValueError("All Nicheformer gene mappings were ambiguous")
    destination = resolve(root, config["gene_mapping_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(destination, sep="\t", index=False)
    return {"path": str(destination), "sha256": file_sha256(destination),
            "gene_ids_path": str(ids_path), "gene_ids_sha256": file_sha256(ids_path),
            "reference_sha256": file_sha256(reference_path), "vocabulary_sha256": file_sha256(vocabulary_path),
            "mapped_genes": len(mapped), "native_mouse_genes": int(mapped.mapping_kind.eq("native_mouse").sum()),
            "human_ortholog_genes": int(mapped.mapping_kind.eq("one_to_one_human").sum()),
            "ambiguous_source_symbols_removed": len(bad_sources), "ambiguous_target_tokens_removed": len(bad_targets),
            "reference_tokens": len(native), "ensembl_version_suffixes_removed": True}
