#!/usr/bin/zsh
#SBATCH --job-name=cellencoder
#SBATCH --account=thes2301
#SBATCH --export=ALL
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/encoder-%j.log
set -euo pipefail
ROOT=${CPP_ROOT:-${SLURM_SUBMIT_DIR:?Set CPP_ROOT or submit from the project directory}}
[[ -f "$ROOT/cluster/run.sh" ]] || { echo "Project directory not found: $ROOT" >&2; exit 2; }
export CPP_ROOT="$ROOT"
export CPP_CONFIG="${CPP_CONFIG:-configs/benchmark.yaml}"
export SLURM_EXPORT_ENV=ALL
ENCODER=${1:?Pass uce or nicheformer}
case "$ENCODER" in uce|nicheformer) export CPP_ENV="$ENCODER";; *) echo 'GPU encoder must be uce or nicheformer' >&2; exit 2;; esac
cd "$ROOT"
srun --export=ALL zsh "$ROOT/cluster/run.sh" embed --encoder "$ENCODER" --device cuda
