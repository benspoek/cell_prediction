#!/usr/bin/zsh
# Dry-run by default. No jobs are submitted without --execute.
set -euo pipefail
ROOT=${CPP_ROOT:?Set CPP_ROOT to the project directory on the cluster}
JOBS=${1:?Usage: zsh cluster/submit.sh jobs.tsv [--execute] [--cpu]}
shift
EXECUTE=0
CPU=0
for ARG in "$@"; do case "$ARG" in --execute) EXECUTE=1;; --cpu) CPU=1;; *) echo "Unknown argument: $ARG" >&2; exit 2;; esac; done
cd "$ROOT"
mkdir -p logs
export CPP_ROOT="$ROOT"
export CPP_CONFIG="${CPP_CONFIG:-configs/benchmark.yaml}"
N=$("$ROOT/.venvs/core/bin/python" - "$JOBS" "$CPU" <<'PY'
import csv,sys
with open(sys.argv[1],newline='') as stream: rows=list(csv.DictReader(stream,delimiter='\t'))
expected={'ridge','mean','sparse_rrr'} if int(sys.argv[2]) else {'tabpfn','limix','kumo'}
if any(row['predictor'] not in expected for row in rows):
    raise SystemExit('Manifest predictors disagree with --cpu/GPU resource selection')
print(len(rows))
PY
)
(( N > 0 )) || { echo 'Empty manifest' >&2; exit 2; }
LIMIT=${CPP_ARRAY_CONCURRENCY:-4}
PARAMS=(--export=ALL --array="0-$((N-1))%$LIMIT")
if (( CPU )); then PARAMS+=(--partition=c23mm --gres=none --time=04:00:00); fi
if (( EXECUTE )); then
    if (( ! CPU )); then zsh cluster/run.sh budget --jobs "$JOBS" --require-within-budget; fi
    sbatch "${PARAMS[@]}" cluster/evaluate.sbatch "$JOBS"
else
    printf 'Dry run: '
    printf '%q ' sbatch "${PARAMS[@]}" cluster/evaluate.sbatch "$JOBS"
    printf '\nJobs: %s; target/fold fits: %s\n' "$N" "$((N*17))"
fi
