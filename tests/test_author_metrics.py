"""Synthetic author-score definition and failure-policy regression tests."""
import copy

import numpy as np
import pandas as pd
import pytest

from cell_property_prediction.author_metrics import authors_joint_scores
from cell_property_prediction.data import LOG_TARGETS


def example():
    names = list(LOG_TARGETS) + ["AP threshold (mV)"]
    z = np.array([[0, .4, .6, -50], [.2, .8, .2, -49], [.6, .7, 1, -45],
                  [.9, 1.4, .5, -60], [1.3, 1.6, 1.8, -30], [1.8, 1.2, 2, -20]])
    estimate = z + np.array([.3, -.1, .2, 2.5]) + np.arange(6)[:, None] * [.04, .08, -.03, .2]
    raw, predicted = z.copy(), estimate.copy()
    raw[:, :3], predicted[:, :3] = np.exp(raw[:, :3]), np.exp(predicted[:, :3])
    ids = [f"cell{i}" for i in range(6)]
    raw = pd.DataFrame(raw, index=ids, columns=names)
    predicted = pd.DataFrame(predicted, index=ids, columns=names)
    splits = [{"train_ids": ids[2:], "test_ids": ids[:2]},
              {"train_ids": ids[:2], "test_ids": ids[2:]}]
    records = [{"fold": fold, "cell_id": cell, "target": target,
                "y_true": raw.at[cell, target], "y_pred": predicted.at[cell, target]}
               for fold, split in enumerate(splits) for cell in split["test_ids"] for target in names]
    return pd.DataFrame(records).sample(frac=1, random_state=42), raw, splits, names, z, estimate


def manual(z, estimated, raw, splits, *, fold_scale=False):
    scores = []
    for split in splits:
        train = raw.index.get_indexer(split["train_ids"])
        test = raw.index.get_indexer(split["test_ids"])
        scale = z[train].std(axis=0, ddof=0) if fold_scale else z.std(axis=0, ddof=0)
        scores.append(1 - (((z[test] - estimated[test]) / scale) ** 2).sum()
                      / (((z[test] - z[train].mean(axis=0)) / scale) ** 2).sum())
    return scores


def test_joint_definition_logs_cohort_population_scale_and_equal_fold_weight():
    frame, raw, splits, names, z, estimated = example()
    result, rows = authors_joint_scores(frame, raw, splits, names)
    expected = manual(z, estimated, raw, splits)
    assert result["authors_joint_r2_status"] == "complete"
    assert result["authors_joint_r2_n_folds"] == 2
    assert result["authors_joint_r2"] == pytest.approx(np.mean(expected))
    assert [row["authors_joint_r2"] for row in rows] == pytest.approx(expected)
    assert all(row["status"] == "complete" and row["n_targets"] == 4 for row in rows)
    assert [row["n_predictions"] for row in rows] == [8, 16]
    assert result["authors_joint_r2"] != pytest.approx(np.average(expected, weights=[2, 4]))
    assert result["authors_joint_r2"] != pytest.approx(np.mean(manual(z, estimated, raw, splits, fold_scale=True)))
    raw_estimated = estimated.copy()
    raw_estimated[:, :3] = np.exp(raw_estimated[:, :3])
    assert result["authors_joint_r2"] != pytest.approx(np.mean(manual(raw.to_numpy(), raw_estimated, raw, splits)))


def test_uses_declared_training_ids_instead_of_test_complement():
    frame, raw, splits, names, z, estimated = example()
    original = np.mean(manual(z, estimated, raw, splits))
    splits[0]["train_ids"] = ["cell2", "cell3"]
    expected = np.mean(manual(z, estimated, raw, splits))
    result, _ = authors_joint_scores(frame, raw, splits, names)
    assert result["authors_joint_r2"] == pytest.approx(expected)
    assert result["authors_joint_r2"] != pytest.approx(original)


@pytest.mark.parametrize("missing", ["fold", "cell", "target"])
def test_incomplete_oof_has_no_overall_score(missing):
    frame, raw, splits, names, _, _ = example()
    if missing == "fold":
        frame = frame[frame.fold != 0]
    elif missing == "cell":
        frame = frame[frame.cell_id != "cell0"]
    else:
        frame = frame.drop(frame[(frame.fold == 0) & (frame.target == names[0])].index[0])
    result, rows = authors_joint_scores(frame, raw, splits, names)
    assert result["authors_joint_r2_status"] == "incomplete_oof"
    assert np.isnan(result["authors_joint_r2"])
    assert result["authors_joint_r2_n_folds"] == 1
    assert rows[1]["status"] == "complete"


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
@pytest.mark.parametrize("source", ["prediction", "observation"])
def test_invalid_log_domains_or_nonfinite_values_remain_missing(value, source):
    frame, raw, splits, names, _, _ = example()
    if source == "prediction":
        frame.loc[(frame.cell_id == "cell0") & (frame.target == names[0]), "y_pred"] = value
    else:
        raw.loc["cell0", names[0]] = value
    result, rows = authors_joint_scores(frame, raw, splits, names)
    assert result["authors_joint_r2_status"] == "invalid"
    assert result["authors_joint_r2_reason"]
    assert np.isnan(result["authors_joint_r2"])


def test_zero_cohort_standard_deviation_is_not_dropped():
    frame, raw, splits, names, _, _ = example()
    raw[names[-1]] = 1
    result, _ = authors_joint_scores(frame, raw, splits, names)
    assert result["authors_joint_r2_status"] == "invalid"
    assert "standard deviation" in result["authors_joint_r2_reason"]


def test_zero_fold_denominator_is_not_silently_averaged_away():
    raw = pd.DataFrame({"a": [0., 2., 1.]}, index=["x", "y", "z"])
    splits = [{"train_ids": ["x", "y"], "test_ids": ["z"]},
              {"train_ids": ["z"], "test_ids": ["x", "y"]}]
    frame = pd.DataFrame([{"fold": i, "cell_id": cell, "target": "a", "y_true": raw.at[cell, "a"],
                           "y_pred": 0.5} for i, split in enumerate(splits) for cell in split["test_ids"]])
    result, rows = authors_joint_scores(frame, raw, splits, ["a"])
    assert result["authors_joint_r2_status"] == "invalid"
    assert np.isnan(result["authors_joint_r2"])
    assert rows[0]["sst"] == 0 and rows[0]["status"] == "invalid"
    assert result["authors_joint_r2_n_folds"] == 1


@pytest.mark.parametrize("kind", ["duplicate", "unknown_cell", "unknown_target", "unknown_fold", "wrong_observation"])
def test_bad_prediction_identity_is_rejected(kind):
    frame, raw, splits, names, _, _ = example()
    if kind == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
    elif kind == "unknown_cell": frame.iloc[0, frame.columns.get_loc("cell_id")] = "unknown"
    elif kind == "unknown_target": frame.iloc[0, frame.columns.get_loc("target")] = "unknown"
    elif kind == "unknown_fold": frame.iloc[0, frame.columns.get_loc("fold")] = 10
    else: frame.iloc[0, frame.columns.get_loc("y_true")] += 1
    result, _ = authors_joint_scores(frame, raw, splits, names)
    assert result["authors_joint_r2_status"] == "invalid"
    assert np.isnan(result["authors_joint_r2"])


def test_inputs_are_unchanged():
    frame, raw, splits, names, _, _ = example()
    old_frame, old_raw, old_splits = frame.copy(deep=True), raw.copy(deep=True), copy.deepcopy(splits)
    authors_joint_scores(frame, raw, splits, names)
    pd.testing.assert_frame_equal(frame, old_frame)
    pd.testing.assert_frame_equal(raw, old_raw)
    assert splits == old_splits
