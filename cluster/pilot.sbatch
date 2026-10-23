#!/usr/bin/zsh
#SBATCH --job-name=cellpilot
#SBATCH --account=thes2301
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/pilot-%j.log
set -euo pipefail
ROOT=${CPP_ROOT:?Set CPP_ROOT before sbatch}
MODEL=${1:?Pass tabpfn, limix or kumo}
shift
case "$MODEL" in tabpfn|limix|kumo) export CPP_ENV="$MODEL";; *) echo 'Unknown pilot model' >&2; exit 2;; esac
cd "$ROOT"
srun zsh "$ROOT/cluster/run.sh" pilot --predictors "$MODEL" "$@"
