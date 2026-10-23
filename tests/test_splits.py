import numpy as np
import pandas as pd
import pytest

from cell_property_prediction.splits import Fold, SplitPlan, load_splits, make_splits, save_splits


def test_corrected_cell_folds_cover_every_cell_once():
    cells = [f"cell{i}" for i in range(1219)]
    plan = make_splits(cells, pd.DataFrame(), "cell_kfold")
    assert sorted(len(f.test_ids) for f in plan.folds) == [121] + [122] * 9
    assert set(c for f in plan.folds for c in f.test_ids) == set(cells)
    assert plan.never_tested_ids == ()
    assert plan == make_splits(cells, pd.DataFrame(), "cell_kfold")


def test_historical_preserves_exact_permutation_and_nine_remainders():
    cells = np.array([f"cell{i}" for i in range(1219)])
    plan = make_splits(cells, pd.DataFrame(), "author_historical")
    shuffled = cells[np.random.RandomState(42).permutation(1219)]
    assert plan.folds[0].test_ids == tuple(shuffled[:121])
    assert plan.folds[0].train_ids == tuple(shuffled[121:])
    assert len({c for f in plan.folds for c in f.test_ids}) == 1210
    assert set(plan.never_tested_ids) == set(shuffled[1210:])
    for fold in plan.folds:
        assert set(plan.never_tested_ids).issubset(fold.train_ids)


def test_mouse_groups_are_isolated_even_with_shuffled_metadata():
    cells = [f"cell{i}" for i in range(30)]
    metadata = pd.DataFrame({"Cell": cells, "Mouse": [f"m{i//2}" for i in range(30)]}).iloc[::-1]
    plan = make_splits(cells, metadata, "mouse_group_kfold")
    groups = metadata.set_index("Cell")["Mouse"]
    for fold in plan.folds:
        assert not set(groups.loc[list(fold.train_ids)]) & set(groups.loc[list(fold.test_ids)])
    assert plan.never_tested_ids == ()


def test_missing_mouse_rejected():
    cells = [f"cell{i}" for i in range(10)]
    metadata = pd.DataFrame({"Mouse": ["m0"] * 9 + [None]}, index=cells)
    with pytest.raises(ValueError, match="nonempty"):
        make_splits(cells, metadata, "mouse_group_kfold", n_splits=2)


def test_split_round_trip_and_tampering(tmp_path):
    cells = [f"cell{i}" for i in range(12)]
    plan = make_splits(cells, pd.DataFrame(), n_splits=3)
    path = tmp_path / "splits.json"
    save_splits(plan, path)
    assert load_splits(path) == plan
    corrupted = SplitPlan("cell_kfold", 42, ("a", "b"),
                          (Fold(("a",), ("a", "b")), Fold(("b",), ("a",))))
    with pytest.raises(ValueError, match="partition"):
        corrupted.validate()
