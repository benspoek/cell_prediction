#!/usr/bin/zsh
#SBATCH --job-name=cellstage
#SBATCH --account=thes2301
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage-%j.log
set -euo pipefail
ROOT=${CPP_ROOT:?Set CPP_ROOT before sbatch}
cd "$ROOT"
srun zsh "$ROOT/cluster/run.sh" "$@"
