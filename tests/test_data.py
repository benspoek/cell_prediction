import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cell_property_prediction.data import (
    LOG_TARGETS,
    PUBLISHED_25_GENES,
    TARGET_COLUMNS,
    SourceTables,
    author_expression,
    load_prepared,
    persist_prepared,
    prepare_cohort,
    read_ion_channel_genes,
    transform_author_targets,
)


def source_fixture():
    cells = ["cell3", "cell1", "cell2", "cell4", "cell5"]
    exon = pd.DataFrame([[10, 20, 0], [30, 10, 1], [1, 4, 2], [8, 2, 3], [5, 6, 1]], index=cells, columns=["A", "B", "C"])
    intron = pd.DataFrame([[1, 3, 0], [2, 4, 0], [0, 3, 1], [0, 1, 2], [1, 1, 1]], index=cells, columns=exon.columns)
    lengths = pd.DataFrame({"GeneID": ["C", "A", "B"], "exon_bp": [50, 100, 200], "intron_bp": [0, 200, 100]})
    metadata = pd.DataFrame({
        "Cell": ["cell1", "cell2", "cell3", "cell4", "cell5"],
        "RNA type": ["typeA", "typeB", "", "typeA", "typeB"],
        "Exclusion reasons": ["", "", "", "outside of M1", ""],
        "Mouse": ["mouse1", "mouse2", "mouse1", "mouse3", "mouse2"],
        "RNA family": ["Pvalb", "Sst", "low quality", "Pvalb", "Sst"],
    })
    ephys = pd.DataFrame(2.0, index=cells, columns=TARGET_COLUMNS)
    ephys["cell id"] = ephys.index
    ephys.loc["cell5", "AP amplitude adaptation index"] = np.nan
    return SourceTables(exon, intron.iloc[::-1, ::-1], lengths, metadata, ephys)


def test_length_normalization_includes_every_gene():
    exon = pd.DataFrame([[100, 200, 300]], index=["cell"], columns=["A", "B", "C"])
    intron = pd.DataFrame([[10, 20, 30]], index=exon.index, columns=exon.columns)
    lengths = pd.DataFrame({"exon_bp": [100, 200, 100], "intron_bp": [100, 200, 0]}, index=exon.columns)
    abundance = np.array([1 + 10 / 100.001, 1 + 20 / 200.001, 3 + 30 / .001])
    expected = np.log2(1 + 1e6 * abundance / abundance.sum())
    actual = author_expression(exon, intron, lengths)
    assert_allclose(actual.iloc[0], expected)
    # A single gene's expression retains its contribution to the full denominator.
    assert actual.loc["cell", "A"] < 10


def test_join_cohort_logs_and_sparse_counts():
    tables = source_fixture()
    cohort = prepare_cohort(tables)
    assert cohort.features.index.tolist() == ["cell1", "cell2"]
    assert cohort.features.columns.tolist() == ["A", "B", "C"]
    assert list(cohort.targets_raw.columns) == list(TARGET_COLUMNS)
    assert cohort.cohort_audit["typed_unexcluded_cells"] == 3
    assert cohort.cohort_audit["missing_target_ids"] == ["cell5"]
    assert cohort.metadata["mouse"].tolist() == ["mouse1", "mouse2"]
    assert cohort.metadata["rna_family"].tolist() == ["Pvalb", "Sst"]
    for column in LOG_TARGETS:
        assert_allclose(cohort.targets_author[column], np.log(2))
    assert_allclose(cohort.targets_author["AP amplitude (mV)"], 2)
    expected_counts = tables.exon_counts.loc[["cell1", "cell2"]] + tables.intron_counts.loc[["cell1", "cell2"]]
    assert_array_equal(cohort.counts.toarray(), expected_counts.to_numpy())


def test_duplicate_cell_ids_rejected():
    tables = source_fixture()
    tables.metadata.loc[0, "Cell"] = "cell2"
    with pytest.raises(ValueError, match="unique"):
        prepare_cohort(tables)


def test_missing_gene_lengths_rejected():
    tables = source_fixture()
    with pytest.raises(ValueError, match="gene-length"):
        author_expression(tables.exon_counts, tables.intron_counts, tables.gene_lengths.iloc[:2])


def test_invalid_author_log_rejected():
    raw = pd.DataFrame(2., index=["cell"], columns=TARGET_COLUMNS)
    raw.loc["cell", LOG_TARGETS[0]] = 0
    with pytest.raises(ValueError, match="positive"):
        transform_author_targets(raw)


def test_persist_has_no_labels_in_encoder_input(tmp_path):
    ad = pytest.importorskip("anndata")
    original = prepare_cohort(source_fixture())
    manifest = persist_prepared(original, tmp_path)
    counts = ad.read_h5ad(tmp_path / "counts.h5ad")
    assert list(counts.obs.columns) == ["species"]
    assert counts.obs["species"].eq("mouse").all()
    assert set(counts.layers) == {"exon_counts", "intron_counts"}
    with np.load(tmp_path / "expression.npz", allow_pickle=False) as saved:
        assert set(saved.files) == {"values", "cell_ids", "gene_names"}
        assert saved["values"].dtype == np.float32
    recovered = load_prepared(tmp_path)
    assert_allclose(recovered.features, original.features, rtol=1e-6)
    assert_allclose(recovered.targets_author, original.targets_author)
    assert_array_equal(recovered.counts.toarray(), original.counts.toarray())
    assert manifest["shape"] == [2, 3]


def test_published_panel_and_ion_symbol_mapping(tmp_path):
    assert len(PUBLISHED_25_GENES) == len(set(PUBLISHED_25_GENES)) == 25
    path = tmp_path / "ion.csv"
    pd.DataFrame({"Approved symbol": ["SCN1A", "KCNIP2", "SCN1A"]}).to_csv(path, index=False)
    assert read_ion_channel_genes(path, ["Gad1", "Scn1a", "Kcnip2"]) == ["Scn1a", "Kcnip2"]
