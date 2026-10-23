#!/usr/bin/zsh
#SBATCH --job-name=cellstage
#SBATCH --account=thes2301
#SBATCH --export=ALL
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage-%j.log
set -euo pipefail
ROOT=${CPP_ROOT:-${SLURM_SUBMIT_DIR:?Set CPP_ROOT or submit from the project directory}}
[[ -f "$ROOT/cluster/run.sh" ]] || { echo "Project directory not found: $ROOT" >&2; exit 2; }
export CPP_ROOT="$ROOT"
export CPP_CONFIG="${CPP_CONFIG:-configs/benchmark.yaml}"
export SLURM_EXPORT_ENV=ALL
cd "$ROOT"
srun --export=ALL zsh "$ROOT/cluster/run.sh" "$@"
