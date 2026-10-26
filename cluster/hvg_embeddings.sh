#!/usr/bin/zsh
# Standalone 180-task addition: native fold HVGs through all three frozen encoders.
set -euo pipefail
ROOT=${CPP_ROOT:?Set CPP_ROOT to the existing project directory on the cluster}
exec zsh "$ROOT/cluster/pending.sh" --hvg-embeddings "$@"
