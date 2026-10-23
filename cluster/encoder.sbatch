#!/usr/bin/zsh
#SBATCH --job-name=cellencoder
#SBATCH --account=thes2301
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/encoder-%j.log
set -euo pipefail
ROOT=${CPP_ROOT:?Set CPP_ROOT before sbatch}
ENCODER=${1:?Pass uce or nicheformer}
case "$ENCODER" in uce|nicheformer) export CPP_ENV="$ENCODER";; *) echo 'GPU encoder must be uce or nicheformer' >&2; exit 2;; esac
cd "$ROOT"
srun zsh "$ROOT/cluster/run.sh" embed --encoder "$ENCODER" --device cuda
