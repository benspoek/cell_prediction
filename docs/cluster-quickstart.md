# Cluster quickstart

This starts the three simple TabPFN 3.5 benchmarks: broad-expression PCA128,
the ion-channel panel and the published 25-gene panel. Both cell-wise and
mouse-held-out CV predict all 17 electrophysiological targets.

## 1. Clone or update

Log in to your cluster account and choose your project/scratch directory. For a
new checkout:

```zsh
git clone https://github.com/benspoek/cell_prediction.git
cd cell_prediction
export CPP_ROOT="$PWD"
```

For an existing checkout instead:

```zsh
cd /your/path/to/cell_prediction
git pull --ff-only
export CPP_ROOT="$PWD"
```

Only update when no pipeline jobs are queued or running: Slurm jobs read the
source files from this directory. Keep local configuration changes outside
tracked files, as described below, so later pulls remain straightforward.

## 2. Set up the environments

Load a cluster module that provides Python 3.12. The exact module name depends
on the cluster's current software installation; inspect `module avail` rather
than guessing a version. Check the selected interpreter:

```zsh
python3.12 --version
# If it has a different executable path, set this before setup:
# export CPP_PYTHON=/absolute/path/to/python3.12
zsh cluster/start.sh setup
```

Setup creates separate `.venvs/core` and `.venvs/tabpfn` environments and downloads
the pinned source data and TabPFN checkpoint. Run it where network downloads are
allowed. If the model download requires authentication, supply your own
`HF_TOKEN` in the environment after accepting its access conditions; do not put
tokens in files tracked by Git. No Slurm jobs are started during setup.

## 3. Submit preparation and the pilot

```zsh
zsh cluster/start.sh pilot
```

This submits three jobs with success dependencies:

```text
prepare -> splits -> TabPFN pilot
```

Preparation and splits use `c23mm` (8 CPUs, 32 GiB, 4 h). The pilot uses `c23g`
(1 GPU, 8 CPUs, 32 GiB host RAM, 6 h). All use account `thes2301`. The pilot
evaluates two targets in the first cell fold for each of the three input panels.
It stops before submitting the full experiment.

Job IDs are saved in `logs/start-pilot-jobs.txt`. Inspect the queue and logs:

```zsh
squeue --me
ls logs/
# Replace JOBID with the pilot job ID:
tail -f logs/pilot-JOBID.log
```

If an earlier job fails, fix its error before resubmitting; dependent jobs do
not become successful automatically. Finished target results can be resumed.

### Recovery from a missing `CPP_ROOT` in a batch job

If a job exits with `CPP_ROOT: Set CPP_ROOT before sbatch`, it stopped before
Python or data preparation started. Update the scripts after the failed chain
has finished; setup and downloads can be reused:

```zsh
cd ~/SMPC/cell_prediction
git pull --ff-only
export CPP_ROOT="$PWD"
# Use this if you already created the local configuration:
export CPP_CONFIG=configs/cluster.local.yaml
# Offline wrapper tests: fake Slurm commands, no jobs or model downloads:
.venvs/core/bin/python -m pytest -q tests/test_cluster_scripts.py
zsh cluster/start.sh pilot
```

The launchers explicitly use `sbatch --export=ALL`; job steps use
`srun --export=ALL`. Batch scripts also recover a missing `CPP_ROOT` from
`SLURM_SUBMIT_DIR` when submitted from the project directory, and validate that
directory before starting. This covers filtered submission environments without
depending on the site's export defaults. A new pilot command submits a fresh
prepare/splits/pilot chain; failed or dependency-cancelled jobs are not resumed.

## 4. Set the actual budget and start the benchmark

After the pilot completes, check project permissions and available computing
time:

```zsh
r_wlm_usage -q -p thes2301
```

Copy the default configuration to an ignored local file and edit its
`cluster.remaining_gpu_hours` value to the actual usable remainder. Account for
already used pilot time, other pending jobs, the GPU/core-hour conversion and
the sliding quota window. The entire nominal project allocation is not the
available remainder.

```zsh
cp configs/benchmark.yaml configs/cluster.local.yaml
nano configs/cluster.local.yaml
export CPP_CONFIG=configs/cluster.local.yaml

# Preview the 60-job array:
zsh cluster/start.sh submit
# Check its estimate against your actual remaining GPU hours:
zsh cluster/run.sh budget --jobs jobs/simple.tsv
# Submit when the check passes:
zsh cluster/start.sh submit --execute
```

Keep `CPP_ROOT` and `CPP_CONFIG` exported when submitting jobs and restore them
after reconnecting. Changing only `remaining_gpu_hours` does not invalidate the
pilot. Do not change model or preprocessing settings between pilot and full
evaluation; those changes require fresh measurements and output directories.

The array runs at most four tasks simultaneously. Each task has one GPU and
handles one representation/fold with 17 target fits. The 60 tasks cover the
three representations and both 10-fold protocols. The budget check uses measured
pilot time with a 2x margin. Its rest-budget input is manual; it does not reserve
quota globally across multiple arrays. Start one production array at a time.

## 5. Generate the report

When the evaluation array has finished successfully:

```zsh
zsh cluster/start.sh report
```

Results appear in `results/reports/`, including per-target scores, OOF predictions,
mouse bootstrap intervals and comparisons. Incomplete OOF methods are not ranked.
Raw data, checkpoints, environments, job manifests and results are excluded from
Git. The full encoder/LimiX/Kumo comparison and statistical baselines are
documented in the repository README.

## 6. Commit later script changes

The launch script is already committed in this repository. If you modify it on
the cluster, review and commit only the intended source files:

```zsh
git diff -- cluster/start.sh
git add cluster/start.sh
git commit -m "Adjust cluster experiment launcher"
git push origin main
```

Pushing from the cluster requires your own GitHub authentication. A model-download
token does not provide GitHub write access. Do not commit generated data, results
or your local configuration containing the remaining allocation.
