# Continue the remaining experiments

Use this workflow in the **existing cluster checkout** after the three simple
TabPFN input/model pairs have completed. It adds the remaining 21 pairs, reuses
the prepared cohort and saved folds, and keeps the original TabPFN predictions.
All real data, model downloads, embedding generation, pilots, and evaluations
stay on the cluster. Local checks use synthetic fixtures and fake Slurm commands.

| Predictor | Inputs in this continuation | Pairs |
|---|---|---:|
| TabPFN 3.5 | GenePT, UCE, Nicheformer, HVG, HVG→PCA128 | 5 |
| LimiX-2 | PCA128, ion channels, authors' 25 genes, GenePT, UCE, Nicheformer, HVG, HVG→PCA128 | 8 |
| Kumo Tabular Large | All eight inputs | 8 |

Each pair has two protocols with ten saved outer folds and 17 targets. The
continuation therefore creates **420 array tasks and 7,140 target/fold fits**.
The three completed pairs already provide six protocol/method rows; the new
pairs add 42, giving 48 rows in the final three-predictor comparison. These counts
exclude the separate ridge, mean, Sparse-RRR, and historical comparisons.

This single production array includes the direct `hvg` and `hvg_pca128` inputs
with every predictor. No separate `cluster/hvg.sh` submission is needed. The
HVG inputs are prepared within each evaluation fold; only the existing three
full-expression encoder jobs are required. HVG→encoder inputs remain a planned
separate benchmark. Reports include both primary raw-scale macro R² and the
secondary authors' joint R².

Do not run `cluster/start.sh setup`, `prepare`, `splits`, or the `simple` stage
again for this continuation. Do not change the prepared data, folds, original
TabPFN resources, or preprocessing/model settings. Update source files only
while pipeline jobs are neither queued nor running: jobs read the checkout
during execution.

## 1. Restore the cluster environment

```zsh
cd /your/existing/path/to/cell_prediction
export CPP_ROOT="$PWD"
# Use the same base configuration as the completed simple experiments:
export CPP_CONFIG=configs/cluster.local.yaml
```

If the original runs used `configs/benchmark.yaml`, retain that configuration
instead. Restore both exported variables after reconnecting. The continuation
uses the same `CPP_CONFIG` for downloads, embeddings, pilots, arrays, and reports.
Only the recorded remaining quota may change between pilot and production.

Load cluster modules that make both Python 3.12 and Python 3.11 available. Inspect
`module avail` for the site's actual module names; there is no assumed module
version. UCE's pinned Torch stack requires Python 3.11. The other new model
environments use Python 3.12. If module names conflict, identify working absolute
interpreter paths and export them explicitly:

```zsh
python3.12 --version
python3.11 --version
# Optional, when the executable names differ:
# export CPP_PYTHON=/absolute/path/to/python3.12
# export CPP_PYTHON_UCE=/absolute/path/to/python3.11
test -x .venvs/core/bin/python
test -x .venvs/tabpfn/bin/python
```

Keep the modules/libraries needed by the existing environments available in batch
jobs. `sbatch --export=ALL` and `srun --export=ALL` carry the selected environment
into the jobs. The wrappers select the core environment for GenePT/reporting,
the respective encoder environment for UCE/Nicheformer, and the predictor's own
environment for each pilot and array task.

### Existing setup completed, no jobs started

For an existing checkout whose `pending.sh setup` has completed, update the
source while no jobs are queued or running, then proceed directly to embeddings
and the expanded pilots. Keep the existing snapshot, environments, resources,
configuration and prepared data. No repeated setup is required:

```zsh
git pull --ff-only
zsh cluster/pending.sh embeddings &&
zsh cluster/pending.sh pilots
```

After the pilots succeed, update the current remaining quota and use the single
`pending.sh submit --execute` command described below. Do not also submit
`cluster/hvg.sh` for this combined run.

## 2. Snapshot the completed work and install only the new models

Run setup on a cluster node where network access is allowed:

```zsh
zsh cluster/pending.sh setup
```

Setup first requires the existing core and TabPFN environments and verifies both
requested Python versions. Before downloading anything, it records and validates
the 60 completed simple fold runs and saves a baseline snapshot under
`results/pending_workflow/`. The snapshot preserves the original config, model
manifest, pilot and reports, and hashes the original raw/prepared data, saved
splits and completed TabPFN results. Subsequent stages check that those original
inputs and results are unchanged. Additional model-manifest entries are allowed;
the original TabPFN entry must remain unchanged.

Setup then runs:

```zsh
CPP_ENV=core zsh cluster/run.sh prefetch --models limix kumo genept uce nicheformer
for model in limix kumo uce nicheformer; do
  zsh cluster/bootstrap.sh "$model"
done
```

It downloads neither the raw data nor TabPFN again, and does not bootstrap core
or TabPFN. Existing model downloads retain their frozen revisions. The new
environment package snapshots appear in `logs/environment-*.lock.txt`. If the
site requires a CUDA wheel index, set its supported `PIP_EXTRA_INDEX_URL` before
setup. No Slurm jobs are submitted by setup.

Model access conditions still apply. Accept required Hugging Face model terms
with your own account and provide `HF_TOKEN` through the cluster environment
where necessary. LimiX-2 has its upstream model-license restrictions. Consult the
linked upstream model/license pages in the repository README. Do not store tokens
in tracked files. Setup may download large encoder checkpoints and resources;
check scratch storage before starting it.

If the initial snapshot fails, inspect its error and the existing run logs.
Missing or incompatible baseline results need investigation; regenerating the
cohort or deleting completed predictions is not the continuation repair path.

## 3. Generate embeddings and queue the pilots

```zsh
zsh cluster/pending.sh embeddings
zsh cluster/pending.sh pilots
```

`embeddings` submits GenePT as a CPU stage job and UCE/Nicheformer as GPU encoder
jobs. `pilots` validates each existing embedding and its audit. For an unfinished
embedding, its saved encoder job must still be active; the TabPFN pilot waits
with `afterok` for those active jobs. Valid completed embeddings can be reused
even after their Slurm job IDs have expired. A missing/invalid embedding without
an active encoder job stops the launcher before it submits pilots.

The three GPU pilots run serially:

```text
unfinished encoders -> TabPFN (5 inputs) -> LimiX (8 inputs) -> Kumo (8 inputs)
```

Each pilot measures two targets in the first saved cell fold. They append to
`results/pending_workflow/pilot_with_hvg/timings.json`; their prediction files are in that
same separate pilot directory. The original `results/pilot/` and previous
`results/pending_workflow/pilot/` remain available. The new directory prevents
old 15-pair timings from being used for the expanded 21-pair matrix.
Serial dependencies prevent multiple pilot jobs from writing the timing JSON at
the same time. All three pilots use the unchanged base compute configuration.

The account and requested resources remain those of the existing wrappers:

| Stage | Partition | Resources | Time |
|---|---|---|---|
| GenePT; final report | `c23mm` | 8 CPUs, 32 GiB RAM, no GPU | 4 h |
| UCE; Nicheformer | `c23g` | 1 GPU, 16 CPUs, 64 GiB host RAM | 4 h |
| Predictor pilot; production array task | `c23g` | 1 GPU, 8 CPUs, 32 GiB host RAM | 6 h |

All use account `thes2301`. Resource requests come from the existing `#SBATCH`
lines, not from automatic YAML translation. The two encoder jobs may overlap;
the pilots run one at a time. Pilot logs establish whether the configured batch
sizes and the allocated GPU memory are suitable before production.

## 4. Inspect the jobs and set the current remaining quota

```zsh
zsh cluster/pending.sh status
squeue --me
# Substitute a saved encoder/pilot job ID:
tail -f logs/encoder-JOBID.log
tail -f logs/pilot-JOBID.log
```

Stage/job IDs are appended to
`results/pending_workflow/slurm-jobs.tsv` and
`logs/pending-workflow-jobs.log`. Status also queries `squeue` and `sacct` when
available. Embedding-readiness diagnostics are saved under the workflow directory.

After every pilot finishes successfully, check the actual project remainder:

```zsh
r_wlm_usage -q -p thes2301
# Edit the existing local config's cluster.remaining_gpu_hours:
nano "$CPP_CONFIG"
```

Use the usable remaining GPU hours after pilot/encoder use, other outstanding
jobs, the site's GPU/core-hour conversion, and its rolling quota window. The
nominal full allocation is not the remaining quota. The launcher cannot query or
reserve this quota automatically. Keep the rest of the compute configuration
unchanged.

## 5. Preview and submit the pending array

```zsh
# Creates jobs/pending.tsv and prints the 420-task submission command:
zsh cluster/pending.sh submit

# Inspect the measured estimate separately if useful:
CPP_ENV=core zsh cluster/run.sh budget --jobs jobs/pending.tsv \
  --pilot results/pending_workflow/pilot_with_hvg/timings.json

# Submits only after the measured pilot/current-quota check passes:
zsh cluster/pending.sh submit --execute
```

The default is a dry run. `--execute` validates all three embeddings and passes
the new timing JSON to `cluster/submit.sh --pilot PATH`; the GPU budget check must
pass before `sbatch` runs. Missing, stale, or incomplete pilot measurements and
an unset/insufficient remaining quota block submission. The estimate uses the
measured pilot with a factor-of-two margin.

The array contains only the 21 pending input/model pairs: it excludes all three
completed simple TabPFN pairs. It runs at most **four GPU tasks at once**, each
handling one input/model/protocol/fold and all 17 targets. The pending launcher
sets the concurrency to four even if a different `CPP_ARRAY_CONCURRENCY` was
exported. Start one production array at a time.

## 6. Report after production succeeds

```zsh
zsh cluster/pending.sh report
```

If the saved production array is active, the CPU report job receives an
`afterok` dependency on the whole array. For a delayed report after the array
has left the queue, the launcher requires `sacct` records confirming that every
one of its 420 tasks completed with exit code `0:0`. It then submits the report
without a dependency on an expired ID. Missing accounting, failed tasks or an
unknown production array stop report submission. Slurm documents the expiry
restriction for successful-job dependencies in
[the sbatch dependency reference](https://slurm.schedmd.com/sbatch.html#OPT_dependency).

The existing reporting stage writes `results/reports/` for the complete
three-predictor matrix, alongside the six preserved protocol/method rows from the
initial experiments. The original reports have already been archived in the
workflow snapshot. Incomplete OOF methods remain unranked; a queued report is
not evidence that the production calculations have finished.

## Recovery and verification

Before setup, embedding resubmission, pilot resubmission or array submission, the
launcher checks saved job IDs with Slurm and rejects changes while conflicting
jobs are queued/running. Queuing pilots behind active encoders and queuing the
report behind active production are supported. A failed prerequisite can leave
dependent jobs pending indefinitely: inspect the saved IDs/logs and cancel the
failed dependency chain before retrying its stage. Do not clear saved IDs to
bypass an active-job check.

After the conflicting jobs are no longer active, retry the relevant stage.
The append-only job history keeps older IDs visible to the checks. Repeating
`pilots` retains completed pilot target timings; repeating production retains
completed target results when their provenance matches. Update the actual
remaining quota before each production retry. Do not run setup between a measured
pilot and production: reinstalling new model environments can invalidate the
measurement. A changed compute setting, source input, embedding or model resource
requires an investigated fresh pilot; provenance errors should not be bypassed
by editing saved fingerprints.

### GenePT ZIP filename error during setup

If setup failed with `GenePT archive must contain one exact model-3 vector pickle`,
pull the fix before retrying. The published Zenodo v2 ZIP names its model-3
payload `GenePT_gene_protein_embedding_model_3_text.pickle.` with a trailing
period; the record description omits it. The loader accepts both exact names,
checks the original published archive MD5, ignores macOS metadata, rejects
ambiguous payloads, and records the actual archive member in provenance.

When `setup && embeddings && pilots` stops at this error, the last two stages
have not run. On the cluster, retry:

```zsh
cd ~/SMPC/cell_prediction
git pull --ff-only
export CPP_ROOT="$PWD"
export CPP_CONFIG=configs/cluster.local.yaml
zsh cluster/pending.sh setup &&
zsh cluster/pending.sh embeddings &&
zsh cluster/pending.sh pilots
```

Keep the same configuration and restore the Python/CUDA modules used by setup.
The cached ZIP is checked and reused, and complete frozen LimiX/Kumo manifest
entries are verified and reused. No data preparation, split generation, cache
removal, or baseline rerun is needed. The remaining UCE/Nicheformer resources
and model environments are installed before encoder/pilot jobs are submitted.
The Hugging Face unauthenticated-request warning is separate from this error.

Run the offline checks on a POSIX host with zsh (for example the cluster core
environment). These tests use fake Slurm commands and do not start jobs or
download models:

```zsh
.venvs/core/bin/python -m pytest -q tests/test_pending_scripts.py tests/test_cluster_scripts.py
zsh -n cluster/pending.sh
zsh -n cluster/submit.sh
```

The GitHub Actions Linux check installs zsh and runs these mocked shell tests
alongside the complete synthetic Python suite. It downloads no scientific data
or model weights and submits no cluster jobs.

The shell tests are skipped locally on Windows and where zsh is unavailable. Synthetic
Python tests for manifests, evaluation and provenance are part of the regular
test suite.
