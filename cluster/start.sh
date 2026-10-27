#!/usr/bin/zsh
# Entry point for the three simple TabPFN benchmarks. Run on the cluster.
set -euo pipefail
ROOT=${CPP_ROOT:?Set CPP_ROOT to the cloned repository directory}
ACTION=${1:-help}
if (( $# )); then shift; fi
cd "$ROOT"
mkdir -p logs
export CPP_ROOT="$ROOT"
export CPP_CONFIG="${CPP_CONFIG:-configs/benchmark.yaml}"
export CPP_ENV=core
case "$ACTION" in
    setup)
        zsh cluster/bootstrap.sh core
        zsh cluster/run.sh prefetch --data --models tabpfn
        zsh cluster/bootstrap.sh tabpfn
        printf 'Setup complete. Next: zsh cluster/start.sh pilot\n'
        ;;
    pilot)
        [[ -x .venvs/core/bin/python && -x .venvs/tabpfn/bin/python ]] || {
            echo 'Run setup first.' >&2; exit 2;
        }
        PREPARE=$(sbatch --export=ALL --parsable cluster/stage.sbatch prepare)
        PREPARE=${PREPARE%%;*}
        SPLITS=$(sbatch --export=ALL --parsable --dependency="afterok:$PREPARE" cluster/stage.sbatch splits --historical)
        SPLITS=${SPLITS%%;*}
        PILOT=$(sbatch --export=ALL --parsable --dependency="afterok:$SPLITS" cluster/pilot.sbatch tabpfn \
            --representations pca128 ion_channels authors25)
        printf 'prepare=%s\nsplits=%s\npilot=%s\n' "$PREPARE" "$SPLITS" "$PILOT" \
            | tee logs/start-pilot-jobs.txt
        printf 'After the pilot succeeds, use submit; executing submission queries the current GPU quota automatically.\n'
        ;;
    submit)
        zsh cluster/run.sh jobs --stage simple --output jobs/simple.tsv
        # submit.sh is a dry run unless the user passes --execute.
        zsh cluster/submit.sh jobs/simple.tsv "$@"
        ;;
    report)
        # Run after the evaluation array finishes. Partial methods remain unranked.
        sbatch --export=ALL cluster/stage.sbatch report
        ;;
    help|-h|--help)
        printf 'Usage: zsh cluster/start.sh {setup|pilot|submit [--execute]|report}\n'
        ;;
    *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
