#!/usr/bin/zsh
#SBATCH --job-name=cellhvgencoder
#SBATCH --account=thes2301
#SBATCH --export=ALL
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/hvg-encoder-%A_%a.log
set -euo pipefail
ROOT=${CPP_ROOT:-${SLURM_SUBMIT_DIR:?Set CPP_ROOT or submit from the project directory}}
[[ -f "$ROOT/cluster/run.sh" ]] || { echo "Project directory not found: $ROOT" >&2; exit 2; }
export CPP_ROOT="$ROOT"
export CPP_CONFIG="${CPP_CONFIG:-configs/benchmark.yaml}"
export SLURM_EXPORT_ENV=ALL
ENCODER=${1:?Pass genept, uce or nicheformer}
case "$ENCODER" in
    genept) export CPP_ENV=core; DEVICE=cpu;;
    uce|nicheformer) export CPP_ENV="$ENCODER"; DEVICE=cuda;;
    *) echo 'Unknown HVG encoder' >&2; exit 2;;
esac
cd "$ROOT"
# Match the configured protocol order and actual saved split lengths.
TASK=$("$ROOT/.venvs/core/bin/python" - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, os, pathlib, sys
from cell_property_prediction.config import load_config, resolve
root = pathlib.Path(os.environ["CPP_ROOT"])
config = load_config(resolve(root, os.environ["CPP_CONFIG"]))
protocols = list(config["evaluation"]["protocols"])
if protocols != ["cell_kfold", "mouse_group_kfold"] or int(config["evaluation"]["n_splits"]) != 10:
    raise SystemExit("HVG encoding requires the configured twenty saved fold tasks")
tasks = []
for protocol in protocols:
    split = json.loads((resolve(root, config["paths"]["splits_dir"]) / f"{protocol}.json").read_text())
    if len(split["folds"]) != 10:
        raise SystemExit("HVG encoding requires exactly ten saved folds per protocol")
    tasks.extend((protocol, fold) for fold in range(len(split["folds"])))
index = int(sys.argv[1])
if index < 0 or index >= len(tasks):
    raise SystemExit("HVG encoder array index is outside saved folds")
print(*tasks[index])
PY
)
TASK_PARTS=(${=TASK})
(( ${#TASK_PARTS} == 2 )) || { echo 'Invalid HVG encoder task' >&2; exit 2; }
srun --export=ALL zsh "$ROOT/cluster/run.sh" embed --gene-set hvg --encoder "$ENCODER" \
    --protocol "$TASK_PARTS[1]" --fold "$TASK_PARTS[2]" --device "$DEVICE"
