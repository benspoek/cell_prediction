"""Scala Patch-seq tables, complete-case cohort, and feature-only exports."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

SOURCE_COMMIT = "3bf98a524b8e48aa8647e1b9ae8f2351d9c712c9"
TARGET_COLUMNS = (
    "AP amplitude (mV)", "AP amplitude adaptation index",
    "AP coefficient of variation", "AP threshold (mV)", "AP width (ms)",
    "Afterhyperpolarization (mV)", "ISI adaptation index",
    "ISI coefficient of variation", "Input resistance (MOhm)", "Latency (ms)",
    "Max number of APs", "Membrane time constant (ms)", "Rebound (mV)",
    "Resting membrane potential (mV)", "Rheobase (pA)", "Sag ratio",
    "Upstroke-to-downstroke ratio",
)
LOG_TARGETS = (
    "AP coefficient of variation", "ISI coefficient of variation",
    "ISI adaptation index",
)
PUBLISHED_25_GENES = (
    "Arpp21", "Bcan", "Bdnf", "Cacna1e", "Cacna2d1", "Cacna2d3", "Coro6",
    "Erbb4", "Gabrd", "Gad1", "Galnt14", "Htr3a", "Kcnip2", "Lamp5",
    "Mybpc1", "Nxph1", "Pde1a", "Ptk2b", "Pvalb", "Slc6a1", "Sst",
    "Synpr", "Tac1", "Tafa1", "Vip",
)
AUTHORS25 = PUBLISHED_25_GENES
PUBLISHED_ION_CHANNEL_25_GENES = (
    "Ano3", "Asic4", "Cacna1e", "Cacna2d1", "Cacna2d2", "Cacna2d3",
    "Cacng3", "Gabra1", "Gabra2", "Gabrd", "Grik1", "Grin3a", "Hcn1",
    "Htr3a", "Itpr1", "Kcna1", "Kcnc1", "Kcnc2", "Kcnf1", "Kcnh4",
    "Kcnh7", "Kcnt2", "Kcnv1", "Scn1a", "Scn3b",
)


@dataclass
class SourceTables:
    """Count frames are cells by genes; metadata/ephys are indexed by cell ID."""
    exon_counts: pd.DataFrame
    intron_counts: pd.DataFrame
    gene_lengths: pd.DataFrame
    metadata: pd.DataFrame
    ephys: pd.DataFrame


@dataclass
class PreparedCohort:
    features: pd.DataFrame
    targets_raw: pd.DataFrame
    targets_author: pd.DataFrame
    metadata: pd.DataFrame
    cohort_audit: dict[str, Any]
    counts: sparse.csr_matrix | None = None
    exon_counts: sparse.csr_matrix | None = None
    intron_counts: sparse.csr_matrix | None = None


def _check_axis(index: pd.Index, description: str) -> None:
    if index.has_duplicates or index.isna().any() or (index.astype(str) == "").any():
        raise ValueError(f"{description} must contain unique nonempty identifiers")


def _cell_index(table: pd.DataFrame, column: str) -> pd.DataFrame:
    result = table.copy()
    if column in result.columns:
        if result[column].isna().any():
            raise ValueError(f"Missing {column} identifiers")
        result.index = pd.Index(result[column].astype(str), name="cell_id")
    else:
        if result.index.isna().any():
            raise ValueError(f"Missing {column} identifiers")
        result.index = result.index.astype(str).rename("cell_id")
    _check_axis(result.index, column)
    return result


def load_source_tables(data_dir: str | Path) -> SourceTables:
    """Load the original main (room-temperature) dataset without gene filtering."""
    data_dir = Path(data_dir)

    def counts(kind: str) -> pd.DataFrame:
        frame = pd.read_csv(data_dir / f"m1_patchseq_{kind}_counts.csv.gz", index_col=0)
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        frame = frame.T
        frame.index.name, frame.columns.name = "cell_id", "gene_symbol"
        return frame

    lengths = pd.read_csv(data_dir / "gene_lengths.txt")
    if "GeneID" not in lengths:
        raise ValueError("gene_lengths.txt requires GeneID, exon_bp and intron_bp")
    lengths.index = pd.Index(lengths["GeneID"].astype(str), name="gene_symbol")
    return SourceTables(
        counts("exon"), counts("intron"), lengths,
        _cell_index(pd.read_csv(data_dir / "m1_patchseq_meta_data.csv", sep="\t"), "Cell"),
        _cell_index(pd.read_csv(data_dir / "m1_patchseq_ephys_features.csv"), "cell id"),
    )


def _align_counts(exon, intron, lengths):
    for label, frame in (("Exon", exon), ("Intron", intron)):
        _check_axis(frame.index, f"{label} cell axis")
        _check_axis(frame.columns, f"{label} gene axis")
        values = frame.to_numpy()
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"{label} counts must be numeric")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{label} counts must be finite and nonnegative")
    if set(exon.index) != set(intron.index) or set(exon.columns) != set(intron.columns):
        raise ValueError("Exon and intron count axes do not match")
    intron = intron.reindex(index=exon.index, columns=exon.columns)
    lengths = lengths.copy()
    if "GeneID" in lengths:
        lengths.index = pd.Index(lengths["GeneID"].astype(str), name="gene_symbol")
    _check_axis(lengths.index, "Gene-length axis")
    if not set(exon.columns).issubset(lengths.index):
        raise ValueError("A gene in the count matrix has no gene-length record")
    lengths = lengths.reindex(exon.columns)
    if not {"exon_bp", "intron_bp"}.issubset(lengths.columns):
        raise ValueError("Gene lengths require exon_bp and intron_bp")
    el = lengths["exon_bp"].to_numpy(dtype=float)
    il = lengths["intron_bp"].to_numpy(dtype=float)
    if not np.isfinite(el).all() or not np.isfinite(il).all() or (el <= 0).any() or (il < 0).any():
        raise ValueError("Exon lengths must be positive; intron lengths nonnegative")
    return exon, intron, lengths


def author_expression(exon: pd.DataFrame, intron: pd.DataFrame, lengths: pd.DataFrame) -> pd.DataFrame:
    """Log2(1+TPM-like abundance), all genes, no across-cell scaling.

    A=exon/exon_bp + intron/(intron_bp+.001). Divide by each cell's sum
    across ALL genes and multiply by 1e6 before log2(1+A).
    """
    exon, intron, lengths = _align_counts(exon, intron, lengths)
    values = exon.to_numpy(dtype=np.float64, copy=True)
    values /= lengths["exon_bp"].to_numpy(dtype=float)
    denominator = lengths["intron_bp"].to_numpy(dtype=float) + 0.001
    for start in range(0, len(exon), 128):
        values[start:start+128] += intron.iloc[start:start+128].to_numpy(dtype=float) / denominator
    totals = values.sum(axis=1)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("Every cell needs positive finite normalized total counts")
    values *= (1_000_000.0 / totals)[:, None]
    values += 1
    np.log2(values, out=values)
    return pd.DataFrame(values, index=exon.index.copy(), columns=exon.columns.copy())


def transform_author_targets(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply natural logs to the three author outcomes; no z-scoring."""
    missing = set(TARGET_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")
    result = raw.loc[:, list(TARGET_COLUMNS)].astype(float).copy()
    for column in LOG_TARGETS:
        values = result[column]
        if (values.notna() & ((values <= 0) | ~np.isfinite(values))).any():
            raise ValueError(f"{column} must be positive for the author log transform")
        result[column] = np.log(values)
    return result


def prepare_cohort(tables: SourceTables, cohort: str = "author1219") -> PreparedCohort:
    """Nonempty RNA type, no exclusion reason, and 17 complete targets.

    The cohort name describes the real dataset; small synthetic fixtures are
    allowed. No extra RNA-type-confidence threshold is imposed.
    """
    if cohort != "author1219":
        raise ValueError("Supported cohort: author1219")
    exon, intron, lengths = _align_counts(tables.exon_counts, tables.intron_counts, tables.gene_lengths)
    metadata = _cell_index(tables.metadata, "Cell")
    ephys = _cell_index(tables.ephys, "cell id")
    if set(metadata.index) != set(exon.index):
        raise ValueError("Metadata and count cell IDs must match exactly")
    if not set(ephys.index).issubset(metadata.index):
        raise ValueError("Electrophysiology contains unknown cell IDs")
    if not {"RNA type", "Exclusion reasons"}.issubset(metadata.columns):
        raise ValueError("Metadata requires RNA type and Exclusion reasons")
    missing = set(TARGET_COLUMNS) - set(ephys.columns)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")
    raw = ephys.reindex(metadata.index).loc[:, list(TARGET_COLUMNS)].astype(float)
    typed = metadata["RNA type"].fillna("").astype(str).str.strip().ne("")
    unexcluded = metadata["Exclusion reasons"].fillna("").astype(str).str.strip().eq("")
    complete = pd.Series(np.isfinite(raw.to_numpy()).all(axis=1), index=metadata.index)
    selected = typed & unexcluded & complete
    cells = metadata.index[selected]
    if len(cells) == 0:
        raise ValueError("No cells satisfy the author1219 criteria")
    raw = raw.loc[cells].copy()
    transformed = transform_author_targets(raw)
    selected_exon, selected_intron = exon.reindex(cells), intron.reindex(cells)
    features = author_expression(selected_exon, selected_intron, lengths)
    features.index.name = raw.index.name = transformed.index.name = "cell_id"
    metadata = metadata.loc[cells].copy()
    for source, canonical in (("Mouse", "mouse"), ("Slice", "slice"), ("RNA family", "rna_family"), ("RNA type", "rna_type")):
        if source in metadata:
            metadata[canonical] = metadata[source]
    audit = {
        "cohort": cohort, "metadata_cells": int(len(typed)),
        "expression_cells": int(len(exon)), "ephys_cells": int(len(ephys)),
        "typed_unexcluded_cells": int((typed & unexcluded).sum()),
        "selected_cells": int(len(cells)), "genes": int(features.shape[1]),
        "expected_historical_cells": 1219, "matches_historical_size": len(cells) == 1219,
        "no_type_ids": typed.index[~typed].tolist(),
        "excluded_ids": unexcluded.index[~unexcluded].tolist(),
        "missing_target_ids": complete.index[typed & unexcluded & ~complete].tolist(),
        "selection": "nonempty RNA type; empty Exclusion reasons; 17 finite targets",
    }
    exon_csr = sparse.csr_matrix(selected_exon.to_numpy())
    intron_csr = sparse.csr_matrix(selected_intron.to_numpy())
    return PreparedCohort(features, raw, transformed, metadata, audit,
                          (exon_csr + intron_csr).tocsr(), exon_csr, intron_csr)


def read_ion_channel_genes(path: str | Path, available_genes=None) -> list[str]:
    """Read approved HGNC symbols; optionally map to available mouse symbols."""
    table = pd.read_csv(path)
    if "Approved symbol" not in table:
        raise ValueError("Ion-channel table requires Approved symbol")
    symbols = list(dict.fromkeys(table["Approved symbol"].dropna().astype(str)))
    if available_genes is None:
        return symbols
    approved = {gene.upper() for gene in symbols}
    return [str(gene) for gene in available_genes if str(gene).upper() in approved]


def persist_prepared(cohort: PreparedCohort, output_dir: str | Path) -> dict[str, Any]:
    """Save targets separately; the H5AD contains raw counts and species only."""
    import anndata as ad

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells, genes = cohort.features.index, cohort.features.columns
    for name, frame in (("raw targets", cohort.targets_raw), ("author targets", cohort.targets_author), ("metadata", cohort.metadata)):
        if not cells.equals(frame.index):
            raise ValueError(f"{name} cell ordering differs from expression")
    if any(x is None for x in (cohort.counts, cohort.exon_counts, cohort.intron_counts)):
        raise ValueError("Raw count matrices are required to persist a cohort")
    np.savez_compressed(output_dir / "expression.npz",
                        values=cohort.features.to_numpy(dtype=np.float32),
                        cell_ids=cells.to_numpy(dtype=str), gene_names=genes.to_numpy(dtype=str))
    annotated = ad.AnnData(
        X=cohort.counts.tocsr(),
        obs=pd.DataFrame({"species": "mouse"}, index=pd.Index(cells.astype(str), name="cell_id")),
        var=pd.DataFrame(index=pd.Index(genes.astype(str), name="gene_symbol")),
        layers={"exon_counts": cohort.exon_counts.tocsr(), "intron_counts": cohort.intron_counts.tocsr()},
    )
    annotated.write_h5ad(output_dir / "counts.h5ad", compression="gzip")
    cohort.targets_raw.to_csv(output_dir / "targets_raw.csv", index_label="cell_id")
    cohort.targets_author.to_csv(output_dir / "targets_author.csv", index_label="cell_id")
    cohort.metadata.to_csv(output_dir / "metadata.csv", index_label="cell_id")
    manifest = {
        "schema_version": 1, "source_commit": SOURCE_COMMIT,
        "shape": list(cohort.features.shape), "cohort_audit": cohort.cohort_audit,
        "targets": list(TARGET_COLUMNS), "natural_log_targets": list(LOG_TARGETS),
        "expression": "log2(1+1e6*(exon/exon_bp+intron/(intron_bp+.001))/full_gene_row_sum)",
        "across_cell_scaling": "none; fit only on training cells in each fold",
        "artifact_roles": {
            "expression.npz": "features only; float32 values, cell_ids, gene_names",
            "counts.h5ad": "features only; raw counts, exon/intron layers, species-only obs",
            "metadata.csv": "evaluation metadata including original RNA types and QC",
            "targets_raw.csv": "labels in physical units",
            "targets_author.csv": "labels with three natural logs; not z-scored",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_prepared(output_dir: str | Path, load_counts: bool = True) -> PreparedCohort:
    """Load a prepared cohort and verify alignment across artifacts."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported prepared-cohort schema")
    with np.load(output_dir / "expression.npz", allow_pickle=False) as bundle:
        features = pd.DataFrame(bundle["values"],
            index=pd.Index(bundle["cell_ids"].astype(str), name="cell_id"),
            columns=pd.Index(bundle["gene_names"].astype(str), name="gene_symbol"))
    _check_axis(features.index, "Prepared cell axis")
    _check_axis(features.columns, "Prepared gene axis")
    frames = {}
    for name in ("targets_raw", "targets_author", "metadata"):
        frame = pd.read_csv(output_dir / f"{name}.csv", index_col=0)
        frame.index = frame.index.astype(str).rename("cell_id")
        if not frame.index.equals(features.index):
            raise ValueError(f"{name} cell ordering differs from expression")
        frames[name] = frame
    for name in ("targets_raw", "targets_author"):
        if list(frames[name].columns) != list(TARGET_COLUMNS):
            raise ValueError(f"{name} must contain the 17 targets in canonical order")
    if list(features.shape) != manifest["shape"]:
        raise ValueError("Expression shape differs from manifest")
    counts = exon = intron = None
    if load_counts:
        import anndata as ad
        annotated = ad.read_h5ad(output_dir / "counts.h5ad")
        if not annotated.obs_names.equals(features.index) or not annotated.var_names.equals(features.columns):
            raise ValueError("Count H5AD axes differ from expression")
        if set(annotated.obs.columns) != {"species"} or not annotated.obs["species"].eq("mouse").all():
            raise ValueError("Count H5AD must contain species-only observation metadata")
        counts = sparse.csr_matrix(annotated.X)
        exon = sparse.csr_matrix(annotated.layers["exon_counts"])
        intron = sparse.csr_matrix(annotated.layers["intron_counts"])
    return PreparedCohort(features, frames["targets_raw"], frames["targets_author"],
                          frames["metadata"], manifest["cohort_audit"], counts, exon, intron)
