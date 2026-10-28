#!/usr/bin/zsh
# Continue the original matrix or add HVG inputs without rerunning completed work.
set -euo pipefail
ROOT=${CPP_ROOT:?Set CPP_ROOT to the existing project directory on the cluster}
PROFILE=pending
if [[ "${1:-}" == --hvg ]]; then PROFILE=hvg; shift
elif [[ "${1:-}" == --hvg-embeddings ]]; then PROFILE=hvg_embeddings; shift; fi
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
HVG_ENCODERS=(genept uce nicheformer)
if [[ "$PROFILE" == hvg ]]; then
    EXPECTED_TASKS=120
    ENCODERS=()
    TABPFN_INPUTS=(hvg hvg_pca128)
    OTHER_INPUTS=(hvg hvg_pca128)
    FETCH_MODELS=(limix kumo)
    BOOTSTRAP_MODELS=(limix kumo)
elif [[ "$PROFILE" == hvg_embeddings ]]; then
    EXPECTED_TASKS=180
    ENCODERS=()
    TABPFN_INPUTS=(hvg_genept hvg_uce hvg_nicheformer)
    OTHER_INPUTS=(hvg_genept hvg_uce hvg_nicheformer)
    FETCH_MODELS=(limix kumo genept uce nicheformer)
    BOOTSTRAP_MODELS=(limix kumo uce nicheformer)
else
    EXPECTED_TASKS=600
    # Preserve both historical timing sets; measure all 30 remaining pairs anew.
    PILOT="$STATE/pilot_with_hvg_encoders/timings.json"
    ENCODERS=(genept uce nicheformer)
    TABPFN_INPUTS=(genept uce nicheformer hvg hvg_pca128 hvg_genept hvg_uce hvg_nicheformer)
    OTHER_INPUTS=(pca128 ion_channels authors25 genept uce nicheformer hvg hvg_pca128 hvg_genept hvg_uce hvg_nicheformer)
    FETCH_MODELS=(limix kumo genept uce nicheformer)
    BOOTSTRAP_MODELS=(limix kumo uce nicheformer)
fi

PILOT_TARGET_ARGS=(--targets 17)
if [[ "$PROFILE" == hvg ]]; then PILOT_TARGET_ARGS=(); fi

usage() {
    if [[ "$PROFILE" == hvg ]]; then
        printf 'Usage: zsh cluster/hvg.sh {setup|pilots|submit [--execute]|report|status}\n'
    else
        printf 'Usage: zsh %s {setup|embeddings|panels|encoder-pilot|encoders|pilot|pilots|submit [--execute]|report|status}\n' "$LAUNCHER"
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

ALL_STAGES=(genept uce nicheformer panels hvg_pilot_genept hvg_pilot_uce hvg_pilot_nicheformer hvg_encoder_genept hvg_encoder_uce hvg_encoder_nicheformer pilot_tabpfn pilot_limix pilot_kumo production report)
DOWNSTREAM_STAGES=(pilot_tabpfn pilot_limix pilot_kumo production report)
HVG_STAGES=(panels hvg_pilot_genept hvg_pilot_uce hvg_pilot_nicheformer hvg_encoder_genept hvg_encoder_uce hvg_encoder_nicheformer)

# Downloads and encoder outputs change the common pilot fingerprint. Keep the
# two workflows serial when they share this checkout and these model resources.
guard_other_workflow() {
    local other own_ids="$IDS"
    for other in results/pending_workflow/slurm-jobs.tsv results/hvg_workflow/slurm-jobs.tsv results/hvg_embeddings_workflow/slurm-jobs.tsv; do
        if [[ "$other" != "$own_ids" && -f "$other" ]]; then
            local IDS="$other"
            guard_active "${ALL_STAGES[@]}"
        fi
    done
}

# Reuse completed artifacts; only active unfinished jobs may become dependencies.
full_dependencies() {
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
}

artifact_dependency() {
    local stage=$1 log=$2
    shift 2
    local valid=0
    if zsh cluster/run.sh "$@" > "$log" 2>&1; then valid=1; fi
    last_job "$stage"
    if [[ -n "$LAST_ID" ]]; then
        queue_ids "$LAST_ID"
        # A final artifact may already validate while its writer still owns
        # GPUs. Depend on every active upstream array to preserve the cap.
        if [[ -n "$ACTIVE_IDS" ]]; then ENCODER_IDS+=("$LAST_ID"); return 0; fi
    fi
    if (( valid )); then return 0; fi
    cat "$log" >&2
    echo "No valid artifact or active $stage job. Inspect logs and rerun its stage." >&2
    exit 2
}

validate_encoder_tasks() {
    "$PY" - "$CPP_CONFIG" <<'PY'
import json, pathlib, sys, yaml
config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
protocols = list(config["evaluation"]["protocols"])
if protocols != ["cell_kfold", "mouse_group_kfold"] or int(config["evaluation"]["n_splits"]) != 10:
    raise SystemExit("HVG encoding requires cell_kfold then mouse_group_kfold, ten folds each")
directory = pathlib.Path(config["paths"]["splits_dir"])
for protocol in protocols:
    split = json.loads((directory / f"{protocol}.json").read_text())
    if len(split["folds"]) != 10:
        raise SystemExit("HVG encoding requires exactly twenty saved fold tasks")
PY
}

submit_hvg_encoder() {
    local encoder=$1 indices=$2 stage=$3
    shift 3
    if [[ "$encoder" == genept ]]; then
        submit_job "$stage" --partition=c23mm --gres=none --cpus-per-task=8 --mem=32G \
            "--array=$indices%4" "$@" cluster/hvg_encoder_array.sbatch "$encoder"
    else
        submit_job "$stage" "--array=$indices%2" "$@" cluster/hvg_encoder_array.sbatch "$encoder"
    fi
}

case "$ACTION" in
    help|-h|--help) usage; exit 0;;
    setup|embeddings|panels|encoder-pilot|encoders|pilot|pilots|report|status)
        (( $# == 0 )) || { usage >&2; exit 2; };;
    submit)
        (( $# == 0 )) || [[ $# == 1 && "$1" == --execute ]] || { usage >&2; exit 2; };;
    *) echo "Unknown action: $ACTION" >&2; usage >&2; exit 2;;
esac
if [[ "$PROFILE" == hvg && "$ACTION" == (embeddings|panels|encoder-pilot|encoders) ]]; then
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
        INTERPRETERS=()
        NEED312=0
        NEED311=0
        for environment in "${BOOTSTRAP_MODELS[@]}"; do
            if [[ ! -x "$ROOT/.venvs/$environment/bin/python" ]]; then
                if [[ "$environment" == uce ]]; then NEED311=1; else NEED312=1; fi
            fi
        done
        if (( NEED312 )); then INTERPRETERS+=("${PY312}:3.12"); fi
        if (( NEED311 )); then INTERPRETERS+=("${PY311}:3.11"); fi
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
            if [[ -x "$ROOT/.venvs/$environment/bin/python" ]]; then continue; fi
            zsh cluster/bootstrap.sh "$environment"
        done
        preserved
        if [[ "$PROFILE" == hvg ]]; then
            printf 'HVG setup complete. Next: zsh cluster/hvg.sh pilots\n'
        elif [[ "$PROFILE" == pending ]]; then
            printf 'Continuation setup complete. Next: zsh cluster/pending.sh embeddings\n'
        else
            printf 'HVG encoder setup complete. Next: zsh cluster/hvg_embeddings.sh panels\n'
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
        for encoder in "${ENCODERS[@]}"; do
            if preserved --embedding "$encoder" > "$STATE/$encoder-readiness.log" 2>&1; then continue; fi
            if [[ "$encoder" == genept ]]; then
                submit_job genept cluster/stage.sbatch embed --encoder genept --device cpu
            else
                submit_job "$encoder" cluster/encoder.sbatch "$encoder"
            fi
        done
        printf 'Encoder IDs saved in %s. Next: zsh cluster/pending.sh panels\n' "$IDS"
        ;;
    panels)
        guard_other_workflow
        guard_active "${HVG_STAGES[@]}" "${DOWNSTREAM_STAGES[@]}"
        preserved
        if zsh cluster/run.sh hvg-panels --check > "$STATE/panels-readiness.log" 2>&1; then
            echo 'All native HVG panels already validate.'
        else
            submit_job panels cluster/stage.sbatch hvg-panels
        fi
        ;;
    encoder-pilot)
        guard_other_workflow
        guard_active hvg_pilot_genept hvg_pilot_uce hvg_pilot_nicheformer hvg_encoder_genept hvg_encoder_uce hvg_encoder_nicheformer "${DOWNSTREAM_STAGES[@]}"
        preserved
        validate_encoder_tasks
        full_dependencies
        artifact_dependency panels "$STATE/panels-readiness.log" hvg-panels --check
        DEPENDENCY=()
        if (( ${#ENCODER_IDS} )); then DEPENDENCY=("--dependency=afterok:${(j.:.)ENCODER_IDS}"); fi
        for encoder in "${HVG_ENCODERS[@]}"; do
            if zsh cluster/run.sh hvg-embedding-check --encoder "$encoder" --first-only > "$STATE/$encoder-first-readiness.log" 2>&1; then continue; fi
            submit_hvg_encoder "$encoder" 0 "hvg_pilot_$encoder" "${DEPENDENCY[@]}"
        done
        echo 'After all three first-fold encoders finish, run encoders; its budget gate queries the current GPU quota automatically.'
        ;;
    encoders)
        guard_other_workflow
        guard_active "${ALL_STAGES[@]}"
        preserved
        validate_encoder_tasks
        zsh cluster/run.sh hvg-panels --check
        zsh cluster/run.sh hvg-embedding-check --first-only
        BUDGET="$STATE/encoder-budget.json"
        # A failed budget command must never leave an approved decision for use.
        zsh cluster/run.sh hvg-encoder-budget --require-within-budget > "$BUDGET"
        cat "$BUDGET"
        for encoder in "${HVG_ENCODERS[@]}"; do
            INDICES=$("$PY" - "$BUDGET" "$encoder" <<'PY'
import json, sys
decision = json.load(open(sys.argv[1]))
if decision.get("approved") is not True or decision.get("within_budget") is not True:
    raise SystemExit("HVG encoder budget has not been approved")
protocols = {"cell_kfold": 0, "mouse_group_kfold": 10}
indices = []
for row in decision["remaining_jobs"]:
    if row["encoder"] != sys.argv[2]:
        continue
    fold = int(row["fold"])
    if row["protocol"] not in protocols or not 0 <= fold < 10:
        raise SystemExit("Invalid remaining HVG encoder task")
    index = protocols[row["protocol"]] + fold
    if index == 0:
        raise SystemExit("The completed pilot cannot appear in the remaining tasks")
    indices.append(index)
print(",".join(map(str, sorted(set(indices)))))
PY
)
            if [[ -z "$INDICES" ]]; then continue; fi
            submit_hvg_encoder "$encoder" "$INDICES" "hvg_encoder_$encoder"
        done
        ;;
    pilot|pilots)
        guard_other_workflow
        guard_active "${DOWNSTREAM_STAGES[@]}"
        preserved
        "$PY" cluster/limix_cache.py --root "$ROOT" --config "$CPP_CONFIG" --apply
        for environment in limix kumo; do
            [[ -x "$ROOT/.venvs/$environment/bin/python" ]] || {
                echo "Missing $environment environment. Run zsh $LAUNCHER setup first." >&2; exit 2;
            }
        done
        full_dependencies
        if [[ "$PROFILE" != hvg ]]; then
            for encoder in "${HVG_ENCODERS[@]}"; do
                artifact_dependency "hvg_encoder_$encoder" "$STATE/$encoder-hvg-readiness.log" hvg-embedding-check --encoder "$encoder"
            done
        fi
        DEPENDENCY=()
        if (( ${#ENCODER_IDS} )); then
            DEPENDENCY=("--dependency=afterok:${(j.:.)ENCODER_IDS}")
        fi
        # Each job updates the same timing JSON, so serialize all three writers.
        submit_job pilot_tabpfn "${DEPENDENCY[@]}" cluster/pilot.sbatch tabpfn \
            --representations "${TABPFN_INPUTS[@]}" "${PILOT_TARGET_ARGS[@]}" --output "$PILOT"
        submit_job pilot_limix "--dependency=afterok:$LAST_JOB" cluster/pilot.sbatch limix \
            --representations "${OTHER_INPUTS[@]}" "${PILOT_TARGET_ARGS[@]}" --output "$PILOT"
        submit_job pilot_kumo "--dependency=afterok:$LAST_JOB" cluster/pilot.sbatch kumo \
            --representations "${OTHER_INPUTS[@]}" "${PILOT_TARGET_ARGS[@]}" --output "$PILOT"
        printf 'After all pilots succeed, use submit --execute; its budget gate queries the current GPU quota automatically.\n'
        ;;
    submit)
        guard_other_workflow
        guard_active "${ALL_STAGES[@]}"
        preserved
        zsh cluster/run.sh jobs --stage "$PROFILE" --output "$JOBS"
        if (( $# )); then
            if [[ "$PROFILE" == pending ]]; then preserved --embeddings; fi
            if [[ "$PROFILE" != hvg ]]; then zsh cluster/run.sh hvg-embedding-check; fi
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
observed = {task for task in rows if task.startswith(f"{job_id}_") and task[len(job_id)+1:].isdigit()}
if observed != expected:
    bad.extend(sorted(observed - expected))
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
