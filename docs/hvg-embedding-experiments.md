# Unified pending benchmark with HVG embeddings

The default pending workflow now includes all remaining expression inputs,
full-expression embeddings, and HVG-derived embeddings in **one 600-task
production array**. It covers 30 input/model pairs, both saved ten-fold protocols,
and 17 targets: **10,200 target/fold fits**. The three completed simple TabPFN
pairs stay excluded. The full fresh-run matrix is 660 tasks.

| Predictor | Pending inputs | Pairs |
|---|---|---:|
| TabPFN 3.5 | GenePT, UCE, Nicheformer, HVG, HVG-PCA128, HVG-GenePT, HVG-UCE, HVG-Nicheformer | 8 |
| LimiX-2 | All eleven inputs, including PCA128, authors25 and ion channels | 11 |
| Kumo Tabular Large | All eleven inputs | 11 |

HVG-GenePT, HVG-UCE and HVG-Nicheformer add nine model/input pairs, 180 prediction
tasks and 18 protocol/input/model summary rows. Complete results across all eleven
inputs and three predictors produce 66 summary rows.

## How the HVG embeddings are constructed

Each saved outer training fold selects approximately 1,000 genes with the exact
existing HVG procedure: the authors' raw-exon detection/mean-expression rule,
followed by removal of genes constant in the training expression matrix. The
same gene list is used by direct HVG, HVG-PCA128 and all three HVG encoders.
Test cells never influence selection.

The encoder input contains raw exon-plus-intron counts for the selected genes,
with species-only observation metadata. Training cells precede test cells in a
fixed order during encoding; final embeddings use canonical cell-ID order.
GenePT retains native panel-based CPM/log1p normalization and gene-count pooling;
UCE retains native count-weighted sampling; Nicheformer retains its frozen gene
mapping, technology means and ranking. Output dimensions remain 3072, 1280 and
512. Encoders remain frozen. This is a native HVG-panel comparison, not a
fixed-normalization gene-removal ablation.

Twenty fold-specific panels and sixty embeddings are shared across the three
predictors. Files live under `embeddings/hvg/{protocol}/fold_XX/`, including the
selection manifest, raw-count H5AD, encoder NPZ files, audits and isolated UCE
working files. Existing full-expression embeddings remain separate and reusable.
Checks bind each output to its exact genes, split, cohort, resources, code and
configuration. Partial, stale or tampered artifacts fail rather than overwrite.

## Pull and prepare

Finish queued/running pipeline jobs before updating source files. All actual
preparation and inference run on the cluster; local tests use synthetic data.
Restore the Python/CUDA modules used by the existing environments, then run:

```zsh
cd ~/SMPC/cell_prediction
git pull --ff-only
export CPP_ROOT="$PWD"
export CPP_CONFIG=configs/cluster.local.yaml
zsh cluster/pending.sh setup
zsh cluster/pending.sh embeddings
zsh cluster/pending.sh panels
zsh cluster/pending.sh encoder-pilot
```

Keep the same base configuration, prepared cohort and split files. Setup verifies
and reuses frozen resources and existing environments. Do not rerun data
preparation, regenerate folds or delete prior results. The embeddings stage
prepares/reuses the three full-expression embeddings. Panels are CPU work. The
encoder pilot uses the first cell fold for GenePT on CPU and UCE/Nicheformer on
GPU; its valid outputs are retained for production.

Use `zsh cluster/pending.sh status` and the saved logs to inspect each stage.
The encoder pilot can depend on active upstream jobs; a failed prerequisite must
be repaired before dependent work proceeds.

## Encoder budget and remaining folds

After all three encoder pilots finish successfully, run the encoder gate. It
queries the current project quota automatically:

```zsh
# Optional: inspect the account, remaining GPU hours, query time and source.
zsh cluster/run.sh quota
zsh cluster/pending.sh encoders
```

Each gate runs `r_wlm_usage -q -p <cluster.account>` afresh and reads
`Consumable gpu-h` from the ML section. The default account is `thes2301`; keep
the same account configuration used for setup and pilots. No quota values need to be
copied into YAML, and the query does not modify configuration or scientific
fingerprints. A failed or unreadable query blocks the gate with an error; it
never silently falls back to an older configured value.

If you explicitly need to use an already configured
`cluster.remaining_gpu_hours`, run
`CPP_QUOTA_SOURCE=config zsh cluster/pending.sh encoders`. The same override can
be used with `submit --execute`; you are responsible for keeping the configured
value current.

The encoder budget uses measured pilot runtimes with a 2x margin for the
remaining fold encodings. It is a separate check from the later predictor
budget. Each encoder pilot already provides its first fold; remaining arrays
cover the other nineteen folds. UCE and Nicheformer each run at most two GPU
tasks concurrently, keeping this phase at four GPUs total. GenePT uses a CPU
array. Existing encoder resources are one GPU, 16 CPUs, 64 GiB host RAM and four
hours; GenePT uses no GPU, eight CPUs and 32 GiB host RAM.

## Full prediction pilot and single production submission

```zsh
zsh cluster/pending.sh pilot
```

`pilot` is an alias for `pilots`. It queues three serial GPU model jobs after the
required embeddings succeed. They exercise **every pending input/model pair on
all seventeen targets in the first saved cell fold**, using production feature
loading, native model adapters and target transforms. This detects missing
resources, invalid dimensions or mappings, failed target transforms and native
inference failures before the production array. It also measures runtime for
budgeting. A pilot cannot guarantee success for every later fold or cluster
node; full artifact validation and resumable target-level results remain active.

Pilot outputs have their own workflow namespace, separate from older 15-pair
and 21-pair timings. The budget fingerprint includes completed fold embeddings
and panels and excludes temporary UCE working files. Source, model or input
changes after the pilot require a fresh measurement.

After all model pilots complete, preview and submit production. `--execute`
queries the usable remaining quota again before submitting:

```zsh
zsh cluster/pending.sh submit
zsh cluster/pending.sh submit --execute
zsh cluster/pending.sh report
```

The first submit command is a preview. Only `--execute` submits the **single
600-task production array**, with at most four simultaneous GPU tasks. Each
uses one GPU, eight CPUs, 32 GiB host RAM and six hours. The measured budget must
cover the array with a 2x margin. The report waits for successful completion of
the whole array, or checks accounting for every task if its job ID has expired.

Reports include primary original-scale per-target R2 and macro R2, paired
mouse-bootstrap comparisons, and secondary authors_joint_r2 with fold-level
SSE/SST diagnostics. The authors' score uses the actual saved splits; it does not
introduce the historical split quirk. Reporting validates the distinct embedding
of every fold and requires one consistent experiment recipe across folds.

## Optional subsets and recovery

`cluster/hvg_embeddings.sh` runs only the three HVG encoders and their 180-task
prediction subset, using the same phases except full-expression `embeddings`.
`cluster/hvg.sh` remains the direct HVG/HVG-PCA128 subset. Neither separate subset
is needed alongside the default unified pending run.

The CLI also supports `embed --gene-set hvg --encoder NAME --protocol PROTOCOL
--fold N`, `hvg-panels`, readiness checks and encoder-budget checks. The default
`embed --gene-set full` path keeps its existing behavior. `jobs --stage
hvg_embeddings` produces the 180-task subset; `jobs --stage pending` produces all
600 remaining tasks.

On failure, inspect logs and saved job IDs. Stop a failed waiting dependency
chain before retrying; do not remove IDs to bypass active-job guards. Matching
completed artifacts and target results are reused. Never clear immutable
baselines or silently replace outputs to repair a provenance error.

### UCE setup download recovery

If setup previously stopped with `tarfile.ReadError: empty file`, pull the fix
and rerun `zsh cluster/pending.sh setup`. No cache deletion is needed. The
resource downloader now retries failed transfers, treats empty cached files as
incomplete, and verifies all five UCE downloads against the published sizes and
MD5 checksums in [Figshare article 24320806 v5](https://api.figshare.com/v2/articles/24320806).
It uses Figshare's canonical download URLs and promotes a temporary download
only after validation. Valid cached assets, existing manifests and experiment
outputs remain intact. A continuing provider/network error stops setup with the
asset path and reason; it never submits inference jobs with invalid resources.

After setup succeeds, continue with `embeddings`, `panels`, and `encoder-pilot`
as shown above. If those commands were entered separately after setup failed,
inspect `zsh cluster/pending.sh status` and the logs before retrying them.

### Nicheformer dissociated reference means

The released 20,310-gene dissociated mean array contains NaNs. Before inference,
setup now follows the authors' [mouse-brain tokenization notebook](https://github.com/theislab/nicheformer/blob/485cadbc5caa15119adfd54228f8a8af835fcabc/notebooks/tokenization/scRNAseq_mouse_brain_haviv.ipynb#L502-L504):
convert NaNs to zero, round half-up, and replace rounded zeros with one. These
are reference normalization denominators, not missing measurements in our cells.

The original array is pinned by its upstream Git blob, retained separately as
`dissociated_mean.raw.npy`, and never overwritten. The prepared denominator array
is saved as `dissociated_mean.npy`; raw and prepared SHA256 hashes, the policy,
NaN count and source are recorded in the resource manifest. Reference length,
infinities and negative values remain checked. Existing frozen resources with a
different policy fail for explicit provenance review. Encoder code and previously
completed simple TabPFN results are unchanged.

For the setup failure `Downloaded Nicheformer means are not a finite nonnegative
array`, pull the fix and rerun `zsh cluster/pending.sh setup`. The legacy raw file
at the prepared path is replaced only if it exactly matches the verified raw
source. Other cached downloads are reused. Then continue with `embeddings`,
`panels`, and `encoder-pilot` (without a trailing `~`).


## Optional GPU-login continuation

For a time-critical run covered by the service desk's login-node advice,
`cluster/login_hvg.py` can replace the queued UCE/Nicheformer HVG encoder
arrays. This is an operational alternative to the existing Slurm phase; it
uses the same saved panels, native environments, resource files and exact
`cuda` device setting. It does not change scientific sources, configuration,
completed outputs or computation fingerprints.

The [RWTH login limits](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/0a23d513f31b4cf1849986aaed475789/)
include four CPUs per user, 30 minutes of GPU access per process, and fair use
of the shared node. The [GPU development guidance](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/a1beccd9e9dc4044a740ed248f478839/)
identifies `login23-g-1`, describes the full-GPU and MIG devices, and recommends
UUID-based device selection. The two first-fold pilots took 136 seconds for
UCE and 69 seconds for Nicheformer. At those rates, the remaining 19 folds per
encoder take about 65 minutes in total when run serially, potentially longer
with four CPUs or shared storage. Ten-minute fold timeouts do not make that
aggregate workload a short development test. Use Slurm when the login-node
allowance does not cover the whole run.

Connect from PowerShell:

```powershell
ssh -o HostName=login23-g-1.hpc.itc.rwth-aachen.de rwth-claix
```

Inspect availability on that node before selecting an available full GPU or
an appropriate MIG instance; host RAM measurements do not establish GPU VRAM
requirements:

```zsh
nvidia-smi -L
nvidia-smi
```

Use the existing tmux session if already inside one. Otherwise start a session
with `tmux new -A -s cellpred`. From the project directory:

```zsh
cd ~/SMPC/cell_prediction
git pull --ff-only
export CPP_ROOT="$PWD"
export CPP_CONFIG=configs/cluster.local.yaml
.venvs/core/bin/python cluster/login_hvg.py
```

That last command is a dry run. It validates existing UCE/Nicheformer artifacts
and lists the missing folds and saved jobs that would be held. It cannot change
Slurm jobs or run an encoder. Set `CPP_LOGIN_GPU` to the **available UUID from
your own output**, replacing the example placeholder below:

```zsh
export CPP_LOGIN_GPU='GPU-REPLACE-WITH-AVAILABLE-UUID'
.venvs/core/bin/python cluster/login_hvg.py --execute --gpu "$CPP_LOGIN_GPU"
```

The execution sequence is:

1. Hold active saved UCE/Nicheformer encoder arrays and pending predictor
   pilots, validating their owner, account, project directory and job name.
   Historical active entries are included to avoid duplicate writers. If an
   encoder task is already running, future tasks stay held and the helper stops;
   wait for the current task to finish and rerun the same command. A running
   predictor or other project job prevents manual execution.
2. Validate existing artifacts and compute only missing folds, sequentially on
   one explicit GPU. The supervisor and its children inherit affinity to at most
   four CPUs. Native thread settings are also capped at four, downloads are
   disabled, and each fold has a ten-minute timeout. Short-lived native probes
   verify CUDA visibility before starting the fold sequence.
3. Validate all 60 HVG embeddings, including GenePT, before canceling the old
   held encoder arrays and pending model pilots. The append-only Slurm job
   history remains intact. Submit replacement prediction pilots with:

```zsh
zsh cluster/pending.sh pilot
```

`--max-folds 1` can be added to the execution command for a one-fold trial. A
successful partial run releases only holds acquired by this helper, allowing
the original Slurm chain to continue and reuse the newly completed artifact.
It does not cancel any job. After all prediction pilots finish, continue with
`zsh cluster/pending.sh submit --execute` as usual.

Progress and process-group IDs are recorded in
`results/pending_workflow/login-hvg.json`; per-fold output is in
`results/pending_workflow/login-hvg-logs/`. The supervisor uses a lock, handles
Ctrl+C/SIGTERM/SIGHUP, and terminates the whole native process group on a timeout
or interruption. On failure, holds and any partial artifacts are preserved.
Resume with the same command after inspecting its log. It refuses to resume or
release holds while a recorded native process group is still alive. A hard
kill or host failure can require manual inspection of that recorded group.
To return jobs to Slurm after the native process has stopped:

```zsh
.venvs/core/bin/python cluster/login_hvg.py --release
```

Do not delete partial immutable artifacts or the saved Slurm history to bypass
a validation or active-writer error. If cancellation has not yet disappeared
from the controller, wait briefly and rerun the helper to confirm completion
before submitting replacement pilots.
