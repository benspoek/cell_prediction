#!/usr/bin/zsh
set -euo pipefail
ROOT=${CPP_ROOT:?Set CPP_ROOT to the project directory on the cluster}
ENVIRONMENT=${1:-core}
PYTHON=${CPP_PYTHON:-python3.12}
if [[ "$ENVIRONMENT" == uce ]]; then PYTHON=${CPP_PYTHON_UCE:-python3.11}; fi
case "$ENVIRONMENT" in core|tabpfn|limix|kumo|uce|nicheformer) ;; *) echo 'Unknown environment' >&2; exit 2;; esac
cd "$ROOT"
command -v "$PYTHON" >/dev/null || { echo "Activate a cluster Python module providing $PYTHON" >&2; exit 2; }
"$PYTHON" -m venv ".venvs/$ENVIRONMENT"
PY="$ROOT/.venvs/$ENVIRONMENT/bin/python"
"$PY" -m pip install --upgrade pip wheel setuptools
if [[ "$ENVIRONMENT" == core ]]; then
    "$PY" -m pip install -e '.[dev]'
else
    "$PY" -m pip install -e '.[dev]' -r "envs/$ENVIRONMENT.txt"
fi
if [[ "$ENVIRONMENT" == limix ]]; then
    [[ -f vendor/LimiX/pyproject.toml ]] || { echo 'Prefetch limix before bootstrap' >&2; exit 2; }
    "$PY" -m pip install -e vendor/LimiX
fi
mkdir -p logs
"$PY" -m pip freeze > "logs/environment-$ENVIRONMENT.lock.txt"
"$PY" -m cell_property_prediction.cli --root "$ROOT" preflight
