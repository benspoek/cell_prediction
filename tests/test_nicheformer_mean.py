"""Synthetic native dissociated-mean preparation; no model/data downloads."""
import shutil

import numpy as np
import pytest

from cell_property_prediction import assets


def test_native_mean_policy_keeps_raw_and_reuses_prepared(tmp_path):
    raw, destination = tmp_path / "raw.npy", tmp_path / "prepared.npy"
    values = np.array([np.nan, 0., .49, .5, 1.49, 1.5, 2.5])
    np.save(raw, values)
    original = raw.read_bytes()
    shutil.copy2(raw, destination)  # The failed setup left raw data at this path.
    audit = assets._prepare_nicheformer_mean(raw, destination, len(values))
    np.testing.assert_array_equal(np.load(destination), [1., 1., 1., 1., 1., 2., 3.])
    assert raw.read_bytes() == original
    assert audit["raw_nan_count"] == 1
    assert audit["policy"] == "dissociated_notebook_half_up_min_one"
    assert audit["raw_sha256"] != audit["prepared_sha256"]
    before = destination.stat().st_mtime_ns
    assert assets._prepare_nicheformer_mean(raw, destination, len(values)) == audit
    assert destination.stat().st_mtime_ns == before


def test_synthetic_released_shape_with_sixty_missing_means(tmp_path):
    raw, destination = tmp_path / "raw.npy", tmp_path / "prepared.npy"
    values = np.full(20310, 2.5)
    values[:60] = np.nan
    np.save(raw, values)
    audit = assets._prepare_nicheformer_mean(raw, destination, 20310)
    prepared = np.load(destination)
    assert audit["raw_nan_count"] == 60
    np.testing.assert_array_equal(prepared[:60], np.ones(60))
    assert np.all(prepared[60:] == 3)
    assert np.isnan(np.load(raw)).sum() == 60


@pytest.mark.parametrize("values,count", [
    ([1., -1.], 2), ([1., np.inf], 2), ([1., -np.inf], 2),
    ([1., 2.], 3), ([[1., 2.]], 2), (1., 1),
])
def test_invalid_raw_mean_never_replaces_existing_file(tmp_path, values, count):
    raw, destination = tmp_path / "raw.npy", tmp_path / "prepared.npy"
    np.save(raw, values)
    destination.write_bytes(b"previous asset")
    with pytest.raises(ValueError):
        assets._prepare_nicheformer_mean(raw, destination, count)
    assert destination.read_bytes() == b"previous asset"


def test_unknown_existing_mean_is_preserved(tmp_path):
    raw, destination = tmp_path / "raw.npy", tmp_path / "prepared.npy"
    np.save(raw, [np.nan, 2.5])
    np.save(destination, [5., 6.])
    original = destination.read_bytes()
    with pytest.raises(ValueError, match="Existing Nicheformer means differ"):
        assets._prepare_nicheformer_mean(raw, destination, 2)
    assert destination.read_bytes() == original
    assert not destination.with_name(destination.name + ".prepared.tmp").exists()
