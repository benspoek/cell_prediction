#!/usr/bin/zsh
# Both author-style HVG inputs, all three foundation models, existing folds.
set -euo pipefail
ROOT=${CPP_ROOT:?Set CPP_ROOT to the existing project directory on the cluster}
exec zsh "$ROOT/cluster/pending.sh" --hvg "$@"
