#!/usr/bin/zsh
# Continue the original matrix or add HVG inputs without rerunning completed work.
set -euo pipefail
ROOT=${CPP_ROOT:?Set CPP_ROOT to the existing project directory on the cluster}
PROFILE=pending
if [[ "${1:-}" == --hvg ]]; then PROFILE=hvg; shift; fi
ACTION=${1:-help}
if (( $# )); then shift; fi
cd "$ROOT"
export CPP_ROOT="$ROOT"
export CPP_CONFIG="${CPP_CONFIG:-configs/benchmark.yaml}"
export CPP_ENV=core
# The continuation runs at most four production GPU tasks concurrently.
export CPP_ARRAY_CONCURRENCY=4
PY="$ROOT/.venvs/core/bin/python"
STATE="results/${PROFILE}_workflow"
SNAPSHOT="$STATE/baseline-snapshot.json"
IDS="$STATE/slurm-jobs.tsv"
PILOT="$STATE/pilot/timings.json"
JOBS="jobs/$PROFILE.tsv"
LAUNCHER="cluster/$PROFILE.sh"
if [[ "$PROFILE" == hvg ]]; then
    EXPECTED_TASKS=120
    ENCODERS=()
    TABPFN_INPUTS=(hvg hvg_pca128)
    OTHER_INPUTS=(hvg hvg_pca128)
    FETCH_MODELS=(limix kumo)
    BOOTSTRAP_MODELS=(limix kumo)
else
    EXPECTED_TASKS=300
    ENCODERS=(genept uce nicheformer)
    TABPFN_INPUTS=(genept uce nicheformer)
    OTHER_INPUTS=(pca128 ion_channels authors25 genept uce nicheformer)
    FETCH_MODELS=(limix kumo genept uce nicheformer)
    BOOTSTRAP_MODELS=(limix kumo uce nicheformer)
fi

usage() {
    if [[ "$PROFILE" == hvg ]]; then
        printf 'Usage: zsh cluster/hvg.sh {setup|pilots|submit [--execute]|report|status}\n'
    else
        printf 'Usage: zsh cluster/pending.sh {setup|embeddings|pilots|submit [--execute]|report|status}\n'
    fi
}

require_existing() {
    [[ -x "$PY" && -x "$ROOT/.venvs/tabpfn/bin/python" ]] || {
        echo 'Continuation requires the existing core and tabpfn environments; do not rerun start.sh setup.' >&2
        exit 2
    }
}

preserved() {
    [[ -f "$SNAPSHOT" ]] || {
        echo "Missing continuation snapshot. Run zsh $LAUNCHER setup first." >&2
        exit 2
    }
    "$PY" cluster/pending_state.py check --root "$ROOT" --config "$CPP_CONFIG" --state "$SNAPSHOT" "$@"
}

last_job() {
    LAST_ID=''
    [[ -f "$IDS" ]] || return 0
    local key id
    while IFS=$'\t' read -r key id; do
        if [[ "$key" == "$1" ]]; then LAST_ID="$id"; fi
    done < "$IDS"
    if [[ -n "$LAST_ID" && "$LAST_ID" != <-> ]]; then
        echo "Invalid saved Slurm job ID for $1: $LAST_ID" >&2; exit 2
    fi
}

remember() {
    [[ "$2" == <-> ]] || { echo "Invalid Slurm job ID: $2" >&2; exit 2; }
    printf '%s\t%s\n' "$1" "$2" >> "$IDS"
    printf '%s=%s\n' "$1" "$2" | tee -a "logs/$PROFILE-workflow-jobs.log"
}

submit_job() {
    local stage=$1
    shift
    LAST_JOB=$(sbatch --export=ALL --parsable "$@")
    LAST_JOB=${LAST_JOB%%;*}
    remember "$stage" "$LAST_JOB"
}

# Keep all recorded IDs in this check: a failed partial submission may still
# leave an older job queued. Encoder jobs are intentionally allowed by pilots.
guard_active() {
    [[ -f "$IDS" ]] || return 0
    local key id wanted
    local -a ids=()
    while IFS=$'\t' read -r key id; do
        for wanted in "$@"; do
            if [[ "$key" == "$wanted" ]]; then
                [[ "$id" == <-> ]] || { echo "Invalid saved Slurm job ID: $id" >&2; exit 2; }
                ids+=("$id")
                break
            fi
        done
    done < "$IDS"
    (( ${#ids} )) || return 0
    ids=("${(@u)ids}")
    queue_ids "${(j:,:)ids}"
    [[ -z "$ACTIVE_IDS" ]] || {
        printf 'Continuation jobs are still queued/running:\n%s\n' "$ACTIVE_IDS" >&2
        echo 'Wait for them, or cancel the failed dependency chain, before retrying this stage.' >&2
        exit 2
    }
}

queue_ids() {
    command -v squeue >/dev/null || { echo 'squeue is required to check saved continuation jobs.' >&2; exit 2; }
    # Filtering the current user's queue locally also works for expired IDs;
    # some sites reject squeue -j when a saved ID has left the controller.
    local queue
    queue=$(squeue -h --me -o '%F|%T') || {
        echo 'Could not check saved jobs; inspect Slurm before retrying this stage.' >&2; exit 2;
    }
    ACTIVE_IDS=$(printf '%s\n' "$queue" | "$PY" -c '
import sys
wanted = set(sys.argv[1].split(","))
seen = set()
for line in sys.stdin.read().splitlines():
    job_id = line.split("|", 1)[0].strip()
    if job_id in wanted:
        seen.add(job_id)
print("\n".join(sorted(seen)))
' "$1")
}

ALL_STAGES=(genept uce nicheformer pilot_tabpfn pilot_limix pilot_kumo production report)
DOWNSTREAM_STAGES=(pilot_tabpfn pilot_limix pilot_kumo production report)

# Downloads and encoder outputs change the common pilot fingerprint. Keep the
# two workflows serial when they share this checkout and these model resources.
guard_other_workflow() {
    local other own_ids="$IDS"
    for other in results/pending_workflow/slurm-jobs.tsv results/hvg_workflow/slurm-jobs.tsv; do
        if [[ "$other" != "$own_ids" && -f "$other" ]]; then
            local IDS="$other"
            guard_active "${ALL_STAGES[@]}"
        fi
    done
}

case "$ACTION" in
    help|-h|--help) usage; exit 0;;
    setup|embeddings|pilots|report|status)
        (( $# == 0 )) || { usage >&2; exit 2; };;
    submit)
        (( $# == 0 )) || [[ $# == 1 && "$1" == --execute ]] || { usage >&2; exit 2; };;
    *) echo "Unknown action: $ACTION" >&2; usage >&2; exit 2;;
esac
if [[ "$PROFILE" == hvg && "$ACTION" == embeddings ]]; then
    echo 'HVG inputs are fitted inside each evaluation fold; no embedding jobs are needed.' >&2
    usage >&2; exit 2
fi
require_existing
mkdir -p logs "$STATE"

case "$ACTION" in
    setup)
        # Check required interpreters before snapshot, downloads, or bootstrap.
        PY312=${CPP_PYTHON:-python3.12}
        PY311=${CPP_PYTHON_UCE:-python3.11}
        INTERPRETERS=("${PY312}:3.12")
        if [[ "$PROFILE" == pending ]]; then INTERPRETERS+=("${PY311}:3.11"); fi
        if [[ "$PROFILE" == hvg && -x "$ROOT/.venvs/limix/bin/python" && -x "$ROOT/.venvs/kumo/bin/python" ]]; then
            INTERPRETERS=()
        fi
        for item in "${INTERPRETERS[@]}"; do
            executable=${item%:*}
            version=${item##*:}
            command -v "$executable" >/dev/null || {
                echo "Load a cluster module providing Python $version, or set CPP_PYTHON/CPP_PYTHON_UCE." >&2
                exit 2
            }
            "$executable" -c 'import sys; expected = tuple(map(int, sys.argv[1].split("."))); assert sys.version_info[:2] == expected, (sys.version, expected)' "$version"
        done
        guard_other_workflow
        guard_active "${ALL_STAGES[@]}"
        "$PY" cluster/pending_state.py snapshot --root "$ROOT" --config "$CPP_CONFIG" --state "$SNAPSHOT"
        # Existing data, TabPFN, core, and tabpfn environments are never rebuilt.
        zsh cluster/run.sh prefetch --models "${FETCH_MODELS[@]}"
        for environment in "${BOOTSTRAP_MODELS[@]}"; do
            if [[ "$PROFILE" == hvg && -x "$ROOT/.venvs/$environment/bin/python" ]]; then continue; fi
            zsh cluster/bootstrap.sh "$environment"
        done
        preserved
        if [[ "$PROFILE" == hvg ]]; then
            printf 'HVG setup complete. Next: zsh cluster/hvg.sh pilots\n'
        else
            printf 'Continuation setup complete. Next: zsh cluster/pending.sh embeddings\n'
        fi
        ;;
    embeddings)
        guard_other_workflow
        guard_active "${ALL_STAGES[@]}"
        preserved
        for environment in uce nicheformer; do
            [[ -x "$ROOT/.venvs/$environment/bin/python" ]] || {
                echo "Missing $environment environment. Run pending.sh setup first." >&2; exit 2;
            }
        done
        submit_job genept cluster/stage.sbatch embed --encoder genept --device cpu
        submit_job uce cluster/encoder.sbatch uce
        submit_job nicheformer cluster/encoder.sbatch nicheformer
        printf 'Encoder IDs saved in %s. Next: zsh cluster/pending.sh pilots\n' "$IDS"
        ;;
    pilots)
        guard_other_workflow
        guard_active "${DOWNSTREAM_STAGES[@]}"
        preserved
        for environment in limix kumo; do
            [[ -x "$ROOT/.venvs/$environment/bin/python" ]] || {
                echo "Missing $environment environment. Run zsh $LAUNCHER setup first." >&2; exit 2;
            }
        done
        ENCODER_IDS=()
        for encoder in "${ENCODERS[@]}"; do
            last_job "$encoder"
            # Completed artifacts can outlive Slurm's controller records. Only
            # add dependencies for unfinished encoders whose jobs remain active.
            if preserved --embedding "$encoder" > "$STATE/$encoder-readiness.log" 2>&1; then
                continue
            fi
            if [[ -z "$LAST_ID" ]]; then
                cat "$STATE/$encoder-readiness.log" >&2
                echo "No valid $encoder embedding or saved encoder job. Run the embeddings stage first." >&2
                exit 2
            fi
            queue_ids "$LAST_ID"
            if ! printf '%s\n' "$ACTIVE_IDS" | "$PY" -c \
                'import sys; raise SystemExit(0 if sys.argv[1] in sys.stdin.read().split() else 1)' "$LAST_ID"; then
                cat "$STATE/$encoder-readiness.log" >&2
                echo "$encoder job $LAST_ID is no longer active and its embedding is invalid. Inspect its log and retry embeddings." >&2
                exit 2
            fi
            ENCODER_IDS+=("$LAST_ID")
        done
        DEPENDENCY=()
        if (( ${#ENCODER_IDS} )); then
            DEPENDENCY=("--dependency=afterok:${(j.:.)ENCODER_IDS}")
        fi
        # Each job updates the same timing JSON, so serialize all three writers.
        submit_job pilot_tabpfn "${DEPENDENCY[@]}" cluster/pilot.sbatch tabpfn \
            --representations "${TABPFN_INPUTS[@]}" --output "$PILOT"
        submit_job pilot_limix "--dependency=afterok:$LAST_JOB" cluster/pilot.sbatch limix \
            --representations "${OTHER_INPUTS[@]}" --output "$PILOT"
        submit_job pilot_kumo "--dependency=afterok:$LAST_JOB" cluster/pilot.sbatch kumo \
            --representations "${OTHER_INPUTS[@]}" --output "$PILOT"
        printf 'After all pilots succeed, update the actual remaining GPU-hour budget before submit --execute.\n'
        ;;
    submit)
        guard_other_workflow
        guard_active "${ALL_STAGES[@]}"
        preserved
        zsh cluster/run.sh jobs --stage "$PROFILE" --output "$JOBS"
        if (( $# )); then
            if [[ "$PROFILE" == pending ]]; then preserved --embeddings; fi
            # submit.sh performs the current-quota check with these pilot timings
            # before sbatch. Retain its normal output for the existing workflow.
            OUTPUT=$(zsh cluster/submit.sh "$JOBS" --pilot "$PILOT" --execute) || {
                printf '%s\n' "$OUTPUT"; exit 2;
            }
            printf '%s\n' "$OUTPUT"
            ARRAY_ID=$(printf '%s\n' "$OUTPUT" | "$PY" -c \
                'import re, sys; matches = re.findall(r"(?m)^Submitted batch job ([0-9]+)\s*$", sys.stdin.read()); assert len(matches) == 1, "Cannot identify submitted array ID"; print(matches[0])')
            remember production "$ARRAY_ID"
        else
            zsh cluster/submit.sh "$JOBS" --pilot "$PILOT"
        fi
        ;;
    report)
        preserved
        guard_active report
        last_job production
        [[ -n "$LAST_ID" ]] || { echo "No saved production array ID; run zsh $LAUNCHER submit --execute first." >&2; exit 2; }
        queue_ids "$LAST_ID"
        if [[ -n "${ACTIVE_IDS//[[:space:]]/}" ]]; then
            submit_job report "--dependency=afterok:$LAST_ID" cluster/stage.sbatch report
        else
            # afterok cannot reference expired controller IDs. For a delayed
            # report, require successful accounting for every production task.
            command -v sacct >/dev/null || { echo 'sacct is required to verify the finished production array.' >&2; exit 2; }
            ACCOUNTING=$(sacct -X --array -n -P -j "$LAST_ID" --format=JobID%100,State%30,ExitCode) || {
                echo 'Could not read production accounting; retry report when sacct is available.' >&2; exit 2;
            }
            printf '%s\n' "$ACCOUNTING" | "$PY" -c '
import sys
job_id = sys.argv[1]
task_count = int(sys.argv[2])
rows = {}
for line in sys.stdin.read().splitlines():
    fields = [field.strip() for field in line.split("|")]
    if len(fields) >= 3:
        rows[fields[0]] = fields[1:3]
expected = {f"{job_id}_{index}" for index in range(task_count)}
bad = [task for task in sorted(expected) if rows.get(task) != ["COMPLETED", "0:0"]]
if bad:
    raise SystemExit(f"Production accounting does not confirm all {task_count} tasks completed with exit 0; missing/failed examples: {bad[:5]}. Inspect sacct and retry the array before report.")
' "$LAST_ID" "$EXPECTED_TASKS"
            submit_job report cluster/stage.sbatch report
        fi
        ;;
    status)
        preserved
        if [[ ! -s "$IDS" ]]; then
            printf 'No continuation jobs have been submitted.\n'
        else
            cat "$IDS"
            JOB_IDS=$("$PY" -c 'import pathlib, sys; print(",".join(dict.fromkeys(line.split("\t")[1] for line in pathlib.Path(sys.argv[1]).read_text().splitlines())))' "$IDS")
            if command -v squeue >/dev/null; then
                queue_ids "$JOB_IDS"
                printf 'Active saved job IDs:\n%s\n' "${ACTIVE_IDS:-none}"
            fi
            if command -v sacct >/dev/null; then
                sacct -j "$JOB_IDS" --format=JobID,JobName,State,Elapsed,AllocTRES
            fi
        fi
        ;;
esac
