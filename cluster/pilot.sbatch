#!/usr/bin/zsh
#SBATCH --job-name=cellpilot
#SBATCH --account=thes2301
#SBATCH --export=ALL
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/pilot-%j.log
set -euo pipefail
ROOT=${CPP_ROOT:-${SLURM_SUBMIT_DIR:?Set CPP_ROOT or submit from the project directory}}
[[ -f "$ROOT/cluster/run.sh" ]] || { echo "Project directory not found: $ROOT" >&2; exit 2; }
export CPP_ROOT="$ROOT"
export CPP_CONFIG="${CPP_CONFIG:-configs/benchmark.yaml}"
export SLURM_EXPORT_ENV=ALL
MODEL=${1:?Pass tabpfn, limix or kumo}
shift
case "$MODEL" in tabpfn|limix|kumo) export CPP_ENV="$MODEL";; *) echo 'Unknown pilot model' >&2; exit 2;; esac
cd "$ROOT"
srun --export=ALL zsh "$ROOT/cluster/run.sh" pilot --predictors "$MODEL" "$@"
