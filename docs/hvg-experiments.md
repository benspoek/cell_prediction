# Standalone HVG subset

This workflow adds two expression inputs for **TabPFN 3.5, LimiX-2, and Kumo
Tabular Large**. It reuses the prepared 1,219-cell cohort, 17 targets, and the
saved ten-fold `cell_kfold` and `mouse_group_kfold` splits.

| Input | Features passed to the predictor |
|---|---|
| `hvg` | Approximately 1,000 author-selected genes, with training-fitted scaling |
| `hvg_pca128` | The same selected genes, scaled and reduced to at most 128 training-fitted principal components |

There are 2 inputs × 3 predictors × 2 protocols × 10 folds = **120 array tasks**,
with 17 target fits each (**2,040 fits**). Both inputs are now included in the
combined `pending` stage (600 tasks) and `full` stage (660 tasks). Use the
[pending-experiment guide](pending-experiments.md) to run all remaining inputs
and HVG together in one production submission, including HVG-derived encoder
embeddings; see the [unified guide](hvg-embedding-experiments.md). This standalone workflow is for
running only the HVG subset; do not submit it alongside the combined pending
array. All eleven inputs across three models produce 66 protocol/input/model
summary rows once the complete comparison finishes.

## Selection rule and interpretation

The [pinned authors' notebook](https://github.com/berenslab/mini-atlas/blob/3bf98a524b8e48aa8647e1b9ae8f2351d9c712c9/code/reduced-rank-regression.ipynb)
calls `rnaseqTools.geneSelection` on raw **exon counts**, targeting 1,000 genes
with `threshold=32` and `decay=1.5`. Its remaining defaults are `atleast=10`,
`yoffset=0.02`, and initial `xoffset=5`. It selects unusually high near-zero
rates relative to mean log expression, adjusting an exponential boundary by
binary search. This is the paper's gene-selection method, not a generic
variance ranking or Scanpy HVG flavor. Ties can prevent an exact count of 1,000;
the actual selected genes and count are recorded.

The notebook selects genes on the entire eligible cohort before CV. This
benchmark instead selects genes from **each training fold only**, then uses
those genes for that fold's training and test cells. Scaling and PCA also use
training cells only. All three predictors and both HVG inputs receive the same
selected gene list for a given fold. Genes can differ between folds.

The feature values use the existing author normalization: length-corrected
exon-plus-intron abundance, normalized per cell over all genes, then
`log2(1 + TPM)`. Selection uses the exon layer of `counts.h5ad`; the combined
count matrix is not substituted. No preparation or split regeneration is needed.
See [methodology](methodology.md) for full definitions and provenance.

The existing `pca128` selects adequately detected, varying genes and fits PCA
across that broader gene set. `hvg_pca128` first applies the authors' approximately
1,000-gene selection, so it is a distinct comparison.

The HVG arrays pass all selected columns to the model adapter. Kumo's existing
native recipe selects at most 500 columns per ensemble member; this behavior
remains recorded in its model audit. Thus the input representation is shared,
while each predictor retains its documented native preprocessing.

## Pull and prepare the model environments

Finish any running pipeline jobs before updating the checkout: batch jobs read
the source files during execution. Run one continuation workflow at a time.
Downloads, embedding generation, or source changes between a pilot and its
production array invalidate pilot provenance. If running the embedding/model
continuation too, finish `cluster/pending.sh` before this workflow, or finish this
workflow first and then start that continuation.

On the cluster:

```zsh
cd ~/SMPC/cell_prediction
git pull --ff-only
export CPP_ROOT="$PWD"
export CPP_CONFIG=configs/cluster.local.yaml

# Restore the Python/CUDA modules needed by your existing environments.
zsh cluster/hvg.sh setup
```

Use the same configuration as the completed simple TabPFN benchmarks. Keep its
compute settings unchanged; only `cluster.remaining_gpu_hours` is updated from
the actual project quota. The new stage does not require editing the configured
representation list, which preserves compatibility with the existing reports.

Setup requires the existing core and TabPFN environments and completed simple
TabPFN results. It creates a separate baseline snapshot in
`results/hvg_workflow/`, verifies original data/folds/results, prefetches LimiX-2
and Kumo resources while reusing frozen downloads, and creates any missing model
environments. Existing core/TabPFN environments are reused. If LimiX/Kumo have
already been installed by `pending.sh setup`, they are reused too. Bootstrap
needs Python 3.12 if a model environment is missing. No encoder environment,
Python 3.11, embedding generation, or raw data download is needed for HVG.

## Measure pilots and submit

```zsh
zsh cluster/hvg.sh pilots
zsh cluster/hvg.sh status
```

Three serial GPU pilot jobs each measure both HVG inputs on two targets in the
first saved cell fold. Their timings and pilot predictions use
`results/hvg_workflow/pilot/`, separately from the original and pending pilots.
Wait for all three pilots to complete, then inspect their logs and update the
current GPU-hour budget:

```zsh
r_wlm_usage -q -p thes2301
nano "$CPP_CONFIG"
# Set cluster.remaining_gpu_hours to the actual remaining GPU hours.

zsh cluster/hvg.sh submit
zsh cluster/hvg.sh submit --execute
zsh cluster/hvg.sh report
```

The first submit command previews the 120-task array. `--execute` requires
matching CUDA pilot measurements for all six input/model pairs and a sufficient
current budget, including the existing 2× timing margin. It records the array
ID in `results/hvg_workflow/slurm-jobs.tsv`. The report can be queued immediately;
it waits for the production array to finish successfully. If the array has
already left the controller, the launcher checks accounting for all 120 tasks
before submitting the report.

Pilots and production tasks use account `thes2301`, partition `c23g`, one GPU,
8 CPUs, 32 GiB host RAM, and at most six hours per task. Production concurrency
is capped at four tasks. The report uses the existing CPU stage wrapper.

## Inspect and retry

```zsh
zsh cluster/hvg.sh status
tail -n 80 logs/pilot-JOB_ID.log
tail -n 80 logs/evaluate-ARRAY_ID_TASK_ID.log
```

The feature audit in each new fold manifest records the selected symbols,
selection parameters, selected-gene count, feature dimension, input hashes, and
PCA explained variance when applicable. Production results are stored under
`results/{cell_kfold,mouse_group_kfold}/{hvg,hvg_pca128}/{tabpfn,limix,kumo}/fold_*`.
The report combines all completed methods into `results/reports/summary.csv`
and the usual target-level and paired-comparison files. HVG adds **12** complete
summary rows; methods from the earlier experiments are retained.

If a stage fails, inspect its log and stop any waiting jobs in the failed
dependency chain before retrying. The launcher blocks duplicate active jobs.
With unchanged inputs/code/configuration, completed target fits and matching
pilot records are reused. Do not regenerate data/folds, edit model settings, or
delete existing predictions to repair a continuation. After a deliberate source
or model change, use a fresh pilot location and resolve stale result contexts
before submitting again; the launcher fails rather than mixing contexts.
