#!/usr/bin/zsh
set -euo pipefail
ROOT=${CPP_ROOT:?Set CPP_ROOT to the project directory on the cluster}
ENVIRONMENT=${CPP_ENV:-core}
PY="$ROOT/.venvs/$ENVIRONMENT/bin/python"
[[ -x "$PY" ]] || { echo "Bootstrap environment first: $ENVIRONMENT" >&2; exit 2; }
cd "$ROOT"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
if [[ ${1:-} != prefetch ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi
# Quota is a launcher decision, outside scientific code and its fingerprints.
case "${1:-}" in
    quota|budget|hvg-encoder-budget)
        exec "$PY" "$ROOT/cluster/quota.py" --root "$ROOT" \
            --config "${CPP_CONFIG:-configs/benchmark.yaml}" "$@"
        ;;
esac
exec "$PY" -m cell_property_prediction.cli --root "$ROOT" --config "${CPP_CONFIG:-configs/benchmark.yaml}" "$@"
