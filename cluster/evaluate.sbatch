#!/usr/bin/zsh
#SBATCH --job-name=cellpredict
#SBATCH --account=thes2301
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/evaluate-%A_%a.log
set -euo pipefail
ROOT=${CPP_ROOT:?Set CPP_ROOT before sbatch}
JOBS=${1:?Pass a job manifest}
cd "$ROOT"
MODEL=$("$ROOT/.venvs/core/bin/python" - "$JOBS" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv,sys
with open(sys.argv[1], newline='') as stream:
    print(list(csv.DictReader(stream, delimiter='\t'))[int(sys.argv[2])]['predictor'])
PY
)
case "$MODEL" in tabpfn|limix|kumo) export CPP_ENV="$MODEL"; DEVICE=cuda;; ridge|mean|sparse_rrr) export CPP_ENV=core; DEVICE=cpu;; *) echo 'Unknown predictor' >&2; exit 2;; esac
srun zsh "$ROOT/cluster/run.sh" array-task --jobs "$JOBS" --device "$DEVICE"
