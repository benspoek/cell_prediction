"""Cell-ID splits with explicit corrected and historical evaluation modes."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


@dataclass(frozen=True)
class Fold:
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


@dataclass(frozen=True)
class SplitPlan:
    scheme: str
    seed: int
    cell_ids: tuple[str, ...]
    folds: tuple[Fold, ...]
    never_tested_ids: tuple[str, ...] = ()
    group_column: str | None = None

    @property
    def n_splits(self) -> int:
        return len(self.folds)

    def validate(self, metadata: pd.DataFrame | None = None) -> None:
        universe = set(self.cell_ids)
        if len(universe) != len(self.cell_ids) or not universe or "" in universe:
            raise ValueError("Split cell IDs must be unique and nonempty")
        if len(self.folds) < 2:
            raise ValueError("At least two folds are required")
        all_test = []
        for fold in self.folds:
            train, test = set(fold.train_ids), set(fold.test_ids)
            if len(train) != len(fold.train_ids) or len(test) != len(fold.test_ids):
                raise ValueError("Duplicate IDs inside a fold")
            if not train or not test or train & test or train | test != universe:
                raise ValueError("Every fold must partition all cells into disjoint nonempty sets")
            all_test.extend(fold.test_ids)
        if len(all_test) != len(set(all_test)):
            raise ValueError("A cell occurs in more than one test fold")
        actual_never = universe - set(all_test)
        if actual_never != set(self.never_tested_ids):
            raise ValueError("never_tested_ids differs from test coverage")
        if self.scheme != "author_historical" and actual_never:
            raise ValueError("Corrected split schemes must test every cell once")
        if self.group_column is not None:
            if metadata is None:
                raise ValueError("Metadata is required to validate group isolation")
            groups = _groups_for_cells(self.cell_ids, metadata, self.group_column)
            for fold in self.folds:
                if set(groups.loc[list(fold.train_ids)]) & set(groups.loc[list(fold.test_ids)]):
                    raise ValueError("A mouse occurs in both training and test sets")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scheme": self.scheme,
            "seed": self.seed,
            "n_splits": self.n_splits,
            "cell_ids": list(self.cell_ids),
            "folds": [
                {"train_ids": list(fold.train_ids), "test_ids": list(fold.test_ids)}
                for fold in self.folds
            ],
            "never_tested_ids": list(self.never_tested_ids),
            "group_column": self.group_column,
        }


def _groups_for_cells(cell_ids: Sequence[str], metadata: pd.DataFrame, group_column: str) -> pd.Series:
    table = metadata.copy()
    if "Cell" in table:
        table.index = table["Cell"].astype(str)
    table.index = table.index.astype(str)
    if table.index.has_duplicates:
        raise ValueError("Metadata cell IDs must be unique")
    if group_column not in table:
        raise ValueError(f"Metadata has no {group_column} grouping column")
    groups = table.reindex(cell_ids)[group_column]
    if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
        raise ValueError("Every cell requires a nonempty mouse group")
    return groups.astype(str)


def make_splits(
    cell_ids: Sequence[str],
    metadata: pd.DataFrame,
    scheme: str = "cell_kfold",
    n_splits: int = 10,
    seed: int = 42,
) -> SplitPlan:
    """Create cell/mouse CV or exact floor-sized historical folds.

    Historical mode uses NumPy RandomState and preserves permuted training
    order. Remainder cells deliberately occur in every training fold. This
    mode exists solely for reproducing the published historical protocol.
    """
    if any(pd.isna(cell) for cell in cell_ids):
        raise ValueError("Cell IDs must be nonmissing")
    cells = np.asarray([str(cell) for cell in cell_ids], dtype=str)
    if len(cells) < n_splits or n_splits < 2:
        raise ValueError("n_splits must be between two and the number of cells")
    if len(set(cells)) != len(cells) or "" in cells:
        raise ValueError("Cell IDs must be unique and nonempty")
    group_column = None
    if scheme == "cell_kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        indices = list(splitter.split(cells))
    elif scheme == "mouse_group_kfold":
        group_column = "Mouse" if "Mouse" in metadata else "mouse"
        groups = _groups_for_cells(cells, metadata, group_column)
        if groups.nunique() < n_splits:
            raise ValueError("GroupKFold requires at least n_splits distinct mice")
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        indices = list(splitter.split(cells, groups=groups.to_numpy()))
    elif scheme == "author_historical":
        permutation = np.random.RandomState(seed).permutation(len(cells))
        block = len(cells) // n_splits
        indices = []
        for fold in range(n_splits):
            test_positions = np.arange(fold * block, (fold + 1) * block)
            train_positions = np.setdiff1d(np.arange(len(cells)), test_positions)
            indices.append((permutation[train_positions], permutation[test_positions]))
    else:
        raise ValueError(f"Unknown split scheme: {scheme}")
    folds = tuple(
        Fold(tuple(cells[train].tolist()), tuple(cells[test].tolist()))
        for train, test in indices
    )
    tested = {cell for fold in folds for cell in fold.test_ids}
    plan = SplitPlan(
        scheme=scheme,
        seed=seed,
        cell_ids=tuple(cells.tolist()),
        folds=folds,
        never_tested_ids=tuple(cell for cell in cells if cell not in tested),
        group_column=group_column,
    )
    plan.validate(metadata)
    return plan


def save_splits(plan: SplitPlan, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_splits(path: str | Path, metadata: pd.DataFrame | None = None) -> SplitPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported split-plan schema")
    plan = SplitPlan(
        scheme=payload["scheme"],
        seed=payload["seed"],
        cell_ids=tuple(payload["cell_ids"]),
        folds=tuple(Fold(tuple(fold["train_ids"]), tuple(fold["test_ids"])) for fold in payload["folds"]),
        never_tested_ids=tuple(payload.get("never_tested_ids", [])),
        group_column=payload.get("group_column"),
    )
    if payload["n_splits"] != plan.n_splits:
        raise ValueError("Split count differs from serialized n_splits")
    if plan.group_column is not None and metadata is None:
        raise ValueError("Loading a mouse split requires metadata to verify isolation")
    plan.validate(metadata)
    return plan
