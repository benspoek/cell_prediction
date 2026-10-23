"""Small contract tests; never download checkpoints or run foundation models."""
from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cell_property_prediction.embeddings import (
    _genept, _nicheformer, _safe_extract_tar, _save, build_nicheformer_gene_mapping, load_genept_vectors,
)
from cell_property_prediction.models import build_predictor


def test_genept_full_library_normalisation_and_original_denominator(tmp_path):
    vectors = np.zeros((1, 3072), dtype=np.float32)
    vectors[0, 0] = 2
    np.savez(tmp_path / "vectors.npz", gene_symbols=np.array(["KCNQ2"]), vectors=vectors)
    (tmp_path / "mapping.tsv").write_text("mouse_symbol\thuman_symbol\nKcnq2\tKCNQ2\n")
    data = SimpleNamespace(X=sparse.csr_matrix([[1, 9], [4, 6]]),
                           n_vars=2, var_names=pd.Index(["Kcnq2", "Unmatched"]))
    config = {"vectors_path": "vectors.npz", "orthologs_path": "mapping.tsv",
              "normalisation_total": 10000, "pooling": "gene_count_mean"}
    values, audit = _genept(data, config, tmp_path)
    # The unrepresented gene still contributes to library size and the /G denominator.
    np.testing.assert_allclose(values[:, 0], np.log1p([1000, 4000]))
    assert audit["matched_genes"] == 1
    np.testing.assert_allclose(audit["retained_raw_count_fraction_min"], 0.1)
    config["pooling"] = "expression_weighted_mean"
    weighted, _ = _genept(data, config, tmp_path)
    np.testing.assert_allclose(weighted[:, 0], 2)


def test_genept_rejects_ambiguous_ortholog_mapping(tmp_path):
    np.savez(tmp_path / "vectors.npz", gene_symbols=np.array(["A"]), vectors=np.zeros((1, 3072)))
    (tmp_path / "mapping.tsv").write_text("mouse_symbol\thuman_symbol\nA1\tA\nA2\tA\n")
    data = SimpleNamespace(X=sparse.csr_matrix([[1, 2]]), n_vars=2, var_names=pd.Index(["A1", "A2"]))
    with pytest.raises(ValueError, match="one-to-one"):
        _genept(data, {"vectors_path": "vectors.npz", "orthologs_path": "mapping.tsv"}, tmp_path)


def test_genept_rejects_ada_vectors_and_preserves_cell_ids(tmp_path):
    np.savez(tmp_path / "wrong.npz", gene_symbols=np.array(["A"]), vectors=np.zeros((1, 1536)))
    with pytest.raises(ValueError, match="3072"):
        load_genept_vectors(tmp_path / "wrong.npz")
    result = _save("fixture", [[2, 3], [4, 5]], ["cell-b", "cell-a"], tmp_path, {})
    with np.load(result.path, allow_pickle=False) as archive:
        assert archive["cell_ids"].tolist() == ["cell-b", "cell-a"]
        assert archive["values"].dtype == np.float32


def test_unsafe_uce_archive_cannot_write_outside_destination(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("../../escaped")
        info.size = 1
        handle.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        _safe_extract_tar(archive, tmp_path / "resources")
    assert not (tmp_path / "escaped").exists()


def test_limiX_no_retrieval_and_no_adaptive_width(tmp_path):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (tmp_path / "weights.ckpt").write_bytes(b"fixture")
    config = {"repo_dir": "vendor", "checkpoint": "weights.ckpt", "inference_config": "infer.json",
              "n_estimators": 1}
    (tmp_path / "infer.json").write_text(json.dumps({"pipelines": [
        {"retrieval_config": {"use_retrieval": True}}]}))
    with pytest.raises(ValueError, match="no retrieval"):
        build_predictor("limix", config, device="cpu", cache_dir=tmp_path / "cache", root=tmp_path)
    (tmp_path / "infer.json").write_text(json.dumps({"pipelines": [{}], "adaptive_svd": {"enabled": True}}))
    with pytest.raises(ValueError, match="adaptive_svd"):
        build_predictor("limix", config, device="cpu", cache_dir=tmp_path / "cache", root=tmp_path)


def test_tabpfn_passes_complete_context_and_explicit_local_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "tabpfn-v3.5-fixture.safetensors"
    checkpoint.write_bytes(b"fixture")
    calls = {}

    class Regressor:
        def __init__(self, **kwargs):
            calls["constructor"] = kwargs
        def fit(self, X, y):
            calls["fit"] = (X.copy(), y.copy())
        def predict(self, X, **kwargs):
            calls["query"] = X.copy()
            calls["predict"] = kwargs
            return np.full(len(X), 3.5)

    monkeypatch.setitem(sys.modules, "tabpfn", SimpleNamespace(TabPFNRegressor=Regressor))
    predictor = build_predictor("tabpfn", {"checkpoint": str(checkpoint), "n_estimators": 8},
                                device="cpu", cache_dir=tmp_path / "cache", root=tmp_path)
    prediction, audit = predictor.predict(np.zeros((11, 3)), np.arange(11.), np.ones((2, 3)), run_id="test")
    np.testing.assert_array_equal(prediction, [3.5, 3.5])
    assert calls["fit"][0].shape == (11, 3)
    assert calls["constructor"]["model_path"] == str(checkpoint)
    assert calls["constructor"]["auto_scale_n_estimators"] is False
    assert calls["predict"]["output_type"] == "mean"
    assert audit["context_rows"] == 11


def test_kumo_uses_large_checkpoint_and_q500_without_context_subsampling(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "large" / "regressor.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fixture")
    calls = {}

    class Table:
        def __init__(self, *, columns, numerical):
            self.columns, self.numerical = columns, numerical
        def __getitem__(self, key):
            index = self.columns["numerical"].index(key)
            return Table(columns={"numerical": [key]}, numerical=self.numerical[:, index:index + 1])

    class Model:
        def __init__(self, **kwargs):
            calls["constructor"] = kwargs
            self.models = {"regression": SimpleNamespace(load_state_dict=self.load)}
        def load(self, state, **kwargs):
            calls["load"] = kwargs
        def to(self, device):
            return self
        def eval(self):
            return self
        def __call__(self, **kwargs):
            calls["context"] = kwargs["x_context"].numerical.shape
            n = len(kwargs["x_query"].numerical)
            return Table(columns={"numerical": [f"q{i:03d}" for i in range(1, 1000)]},
                         numerical=torch.arange(999, dtype=torch.float32).repeat(n, 1))

    monkeypatch.setitem(sys.modules, "sdm", SimpleNamespace(
        models=SimpleNamespace(KumoTabular=Model), Task=SimpleNamespace(regression="regression"),
        Stype=SimpleNamespace(numerical="numerical"), TableTensor=Table))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    predictor = build_predictor("kumo", {"checkpoint": str(checkpoint)}, device="cpu",
                                cache_dir=tmp_path / "cache", root=tmp_path)
    prediction, audit = predictor.predict(np.ones((11, 512)), np.arange(11.), np.ones((2, 512)), run_id="test")
    np.testing.assert_array_equal(prediction, [499, 499])
    assert calls["constructor"]["size"] == "large"
    assert calls["constructor"]["pretrained"] is False
    assert calls["context"] == (11, 512)
    assert audit["native_column_selection_per_estimator"] == 500


def test_nicheformer_mapping_prefers_native_and_drops_ambiguous_tokens(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.h5ad").write_bytes(b"reference")
    genes = ["ENSM1", "ENSG2", "ENSG3", "ENSG4"]
    (checkpoint / "vocab.json").write_text(json.dumps({gene: i + 30 for i, gene in enumerate(genes)}))
    reference = SimpleNamespace(var_names=pd.Index(genes), file=SimpleNamespace(close=lambda: None))
    monkeypatch.setitem(sys.modules, "anndata", SimpleNamespace(read_h5ad=lambda *a, **kw: reference))
    (tmp_path / "ids.tsv").write_text(
        "mouse_symbol\tmouse_ensembl\thuman_ensembl\torthology_type\n"
        "Native\tENSM1.4\tENSG2\tortholog_one2one\n"
        "Human\tENSMmissing\tENSG2.1\tortholog_one2one\n"
        "Ambiguous1\tX\tENSG3\tortholog_one2one\n"
        "Ambiguous2\tY\tENSG3\tortholog_one2one\n"
        "NotOneToOne\tZ\tENSG4\tortholog_one2many\n")
    audit = build_nicheformer_gene_mapping(
        {"checkpoint_dir": "checkpoint", "gene_ids_path": "ids.tsv", "gene_mapping_path": "mapping.tsv"},
        root=tmp_path)
    mapping = pd.read_csv(tmp_path / "mapping.tsv", sep="\t")
    assert dict(zip(mapping.mouse_symbol, mapping.model_gene_id)) == {"Human": "ENSG2", "Native": "ENSM1"}
    assert audit["ambiguous_target_tokens_removed"] == 1


def test_nicheformer_fixed_gene_order_means_and_metadata_tokens(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "model.safetensors", "model.h5ad", "configuration_nicheformer.py",
                 "modeling_nicheformer.py"):
        (checkpoint / name).write_bytes(b"fixture")
    (checkpoint / "vocab.json").write_text(json.dumps({"ENSM1": 30, "ENSG2": 31}))
    np.save(tmp_path / "means.npy", [1000., 0.1])
    (tmp_path / "mapping.tsv").write_text("mouse_symbol\tmodel_gene_id\nA\tENSM1\nB\tENSG2\n")
    reference = SimpleNamespace(var_names=pd.Index(["ENSM1", "ENSG2"]))
    monkeypatch.setitem(sys.modules, "anndata", SimpleNamespace(read_h5ad=lambda *a, **kw: reference))
    calls = {}

    class Model:
        def to(self, device):
            return self
        def eval(self):
            return self
        def get_embeddings(self, **kwargs):
            calls["tokens"] = kwargs["input_ids"].clone()
            calls["layer"] = kwargs["layer"]
            calls["with_context"] = kwargs["with_context"]
            return torch.ones((len(kwargs["input_ids"]), 512))

    loader = SimpleNamespace(from_pretrained=lambda *a, **kw: (Model(), {}))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForMaskedLM=loader))
    data = SimpleNamespace(X=sparse.csr_matrix([[1., 5.], [2., 8.]]), n_obs=2, n_vars=2,
                           var_names=pd.Index(["B", "A"]))
    values, audit = _nicheformer(data, {"checkpoint_dir": "checkpoint", "technology_mean_path": "means.npy",
                                       "gene_mapping_path": "mapping.tsv", "assay_token": 18},
                                 tmp_path, "cpu", True)
    assert values.shape == (2, 512)
    assert calls["tokens"][:, :3].tolist() == [[6, 18, 3], [6, 18, 3]]
    assert calls["tokens"][:, 3].tolist() == [31, 31]  # Low reference mean promotes B over A.
    assert calls["layer"] == -1 and calls["with_context"] is False
    assert audit["matched_genes"] == 2
