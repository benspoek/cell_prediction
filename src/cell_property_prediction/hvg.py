"""Training-fold HVGs using the mini-atlas zero-rate/mean-expression boundary."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import file_sha256
from .rrr import gene_selection

HVG_REPRESENTATIONS = {"hvg", "hvg_pca128"}
HVG_PARAMETERS = {"n": 1000, "threshold": 32, "atleast": 10,
                  "decay": 1.5, "yoffset": .02, "xoffset": 5}
AUTHOR_SOURCE = ("https://github.com/berenslab/mini-atlas/blob/"
                 "3bf98a524b8e48aa8647e1b9ae8f2351d9c712c9/code/reduced-rank-regression.ipynb")


def _load_exon_counts(path: Path, cell_ids: np.ndarray, genes: np.ndarray) -> sparse.csr_matrix:
    """Read the exon layer and align both axes explicitly to expression.npz."""
    import anndata as ad

    annotated = ad.read_h5ad(path)
    if "exon_counts" not in annotated.layers:
        raise ValueError(f"HVG selection requires the raw exon_counts layer: {path}")
    count_cells = annotated.obs_names.to_numpy(dtype=str)
    count_genes = annotated.var_names.to_numpy(dtype=str)
    if len(set(count_cells)) != len(count_cells) or len(set(count_genes)) != len(count_genes):
        raise ValueError("counts.h5ad has duplicate cell or gene identifiers")
    if set(count_cells) != set(cell_ids) or set(count_genes) != set(genes):
        raise ValueError("counts.h5ad cell/gene identifiers differ from expression.npz")
    cells = {name: index for index, name in enumerate(count_cells)}
    columns = {name: index for index, name in enumerate(count_genes)}
    counts = sparse.csr_matrix(annotated.layers["exon_counts"])
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("Raw exon counts must be finite and nonnegative")
    return counts[[cells[cell] for cell in cell_ids]][:, [columns[gene] for gene in genes]].tocsr()


def build_features(config: dict, prepared: Path, representation: str,
                   train_ids: list[str], test_ids: list[str]):
    """Select HVGs on training exons, then scale author-normalized expression."""
    if representation not in HVG_REPRESENTATIONS:
        raise ValueError(f"Unknown HVG representation: {representation}")
    source, counts_path = prepared / "expression.npz", prepared / "counts.h5ad"
    if not counts_path.is_file():
        raise FileNotFoundError(f"HVG selection requires prepared raw exon counts: {counts_path}")
    with np.load(source, allow_pickle=False) as archive:
        values = archive["values"].astype(np.float32)
        cell_ids = archive["cell_ids"].astype(str)
        genes = archive["gene_names"].astype(str)
    if (values.ndim != 2 or values.shape != (len(cell_ids), len(genes))
            or len(set(cell_ids)) != len(cell_ids) or len(set(genes)) != len(genes)):
        raise ValueError("expression.npz has invalid dimensions or duplicate cell/gene identifiers")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Author log-expression must contain finite nonnegative values")
    if (not train_ids or not test_ids or len(set(train_ids)) != len(train_ids)
            or len(set(test_ids)) != len(test_ids) or set(train_ids) & set(test_ids)):
        raise ValueError("HVG features require nonempty disjoint unique train and test cells")
    index = {cell: position for position, cell in enumerate(cell_ids)}
    missing = set(train_ids + test_ids) - index.keys()
    if missing:
        raise ValueError(f"Expression data lack {len(missing)} split cells")
    train_positions = [index[cell] for cell in train_ids]
    test_positions = [index[cell] for cell in test_ids]
    counts = _load_exon_counts(counts_path, cell_ids, genes)
    selected = gene_selection(counts[train_positions], **HVG_PARAMETERS)
    train_raw, test_raw = values[train_positions], values[test_positions]
    varying = np.ptp(train_raw, axis=0) > 0
    keep = selected & varying
    if not keep.any():
        raise ValueError("No HVGs pass the author zero-rate selection and training-fold variance filter")
    scaler = StandardScaler().fit(train_raw[:, keep])
    train, query = scaler.transform(train_raw[:, keep]), scaler.transform(test_raw[:, keep])
    audit = {
        "kind": "training_fold_hvg_expression", "selection_method": "mini_atlas_zero_rate_mean_log_boundary",
        "selection_source": AUTHOR_SOURCE, "selection_parameters": dict(HVG_PARAMETERS),
        "selection_count_kind": "raw exon_counts", "fit_scope": "training_cells_only", "labels_used": False,
        "n_boundary_selected_genes": int(selected.sum()), "n_selected_genes": int(keep.sum()),
        "genes": genes[keep].tolist(), "zero_variance_genes_removed": genes[selected & ~varying].tolist(),
        "expression_transform": "prepared author log2(1+length-corrected TPM); train-only gene scaling",
        "gene_scaler_mean": scaler.mean_.tolist(), "gene_scaler_scale": scaler.scale_.tolist(),
    }
    if representation == "hvg_pca128":
        components = min(128, train.shape[0] - 1, train.shape[1])
        if components < 1:
            raise ValueError("PCA requires at least two training cells")
        pca = PCA(n_components=components, random_state=int(config.get("seed", 42)), svd_solver="randomized")
        train, query = pca.fit_transform(train), pca.transform(query)
        audit.update({"pca_requested_components": 128, "pca_components": components,
                      "explained_variance_ratio": pca.explained_variance_ratio_.tolist()})
    audit["n_features"] = train.shape[1]
    inputs = {str(source): file_sha256(source), str(counts_path): file_sha256(counts_path)}
    return train.astype(np.float32), query.astype(np.float32), audit, inputs
