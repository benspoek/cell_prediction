import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from cell_property_prediction.rrr import fit_rrr, gene_selection, historical_joint_r2


def test_ridge_full_rank_matches_closed_form_and_retains_penalty():
    rng = np.random.RandomState(8)
    x, y = rng.normal(size=(40, 6)), rng.normal(size=(40, 3))
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    penalty = .3
    model = fit_rrr(x, y, rank=3, lambdau=penalty, alpha=0)
    coefficients = np.linalg.solve(x.T @ x + len(x) * penalty * np.eye(6), x.T @ y)
    assert_allclose(model.predict(x), x @ coefficients, rtol=1e-10, atol=1e-10)
    unpenalized = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    assert not np.allclose(model.predict(x), unpenalized)


def test_group_selection_and_relaxation_use_training_features_only():
    rng = np.random.RandomState(1)
    x = rng.normal(size=(200, 8))
    x -= x.mean(axis=0)
    y = x[:, :2] @ np.array([[2., 1.], [-1., 3.]]) + rng.normal(scale=.01, size=(200, 2))
    y -= y.mean(axis=0)
    sparse_fit = fit_rrr(x, y, rank=2, lambdau=.1, relax=False)
    relaxed = fit_rrr(x, y, rank=2, lambdau=.1, relax=True)
    assert sparse_fit.selected[:2].all()
    assert_array_equal(sparse_fit.selected, relaxed.selected)
    assert_array_equal(relaxed.wx[~relaxed.selected], 0)
    assert np.mean((y - relaxed.predict(x)) ** 2) < .2


def test_zero_rate_selection_matches_dense_and_sparse_and_raw_threshold():
    counts = np.zeros((30, 4))
    counts[:15, 0] = 1000
    counts[:10, 1] = 32  # must be strictly greater than 32
    counts[:9, 2] = 1000  # insufficient detection
    counts[:, 3] = 1000  # not excess-zero variable
    dense = gene_selection(counts, n=1)
    assert_array_equal(dense, [True, False, False, False])
    assert_array_equal(dense, gene_selection(sparse.csr_matrix(counts), n=1))


def test_historical_metric_is_joint_training_centered_not_per_target_r2():
    y = np.array([[2., 4.], [4., 8.]])
    prediction = np.zeros_like(y)
    assert historical_joint_r2(y, prediction) == 0
    assert historical_joint_r2(y, y) == 1
    assert np.isnan(historical_joint_r2(np.zeros((2, 2)), np.zeros((2, 2))))


def test_original_backend_never_falls_back(tmp_path):
    with pytest.raises(FileNotFoundError, match="Original backend"):
        fit_rrr(np.ones((5, 2)), np.ones((5, 1)), backend="original",
                original_path=tmp_path / "absent.py")
