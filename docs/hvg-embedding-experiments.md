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

After all three encoder pilots finish successfully, refresh the project quota:

```zsh
r_wlm_usage -q -p thes2301
nano "$CPP_CONFIG"
# Set cluster.remaining_gpu_hours to the actual remaining GPU hours.
zsh cluster/pending.sh encoders
```

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

After all model pilots complete, refresh the usable remaining quota again:

```zsh
r_wlm_usage -q -p thes2301
nano "$CPP_CONFIG"
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
