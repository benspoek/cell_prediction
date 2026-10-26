# Unified pending experiments

The default pending workflow now includes all existing pending inputs, direct
HVG, HVG-PCA128 and HVG-GenePT/UCE/Nicheformer with all three predictors in
**one 600-task production array (10,200 target fits)**. The three completed
simple TabPFN pairs remain excluded. The full fresh-run matrix is 660 tasks.

Follow the [unified HVG-encoder guide](hvg-embedding-experiments.md) for the
complete setup, preparation, pilot, budgeting, submission and recovery sequence.
Do not separately submit `hvg.sh` or `hvg_embeddings.sh` for this combined run.

| Predictor | Pending input/model pairs |
|---|---:|
| TabPFN 3.5 | 8 |
| LimiX-2 | 11 |
| Kumo Tabular Large | 11 |

Each pair uses both saved ten-fold protocols and all 17 targets. Complete
results, including the completed simple benchmarks, yield 66 summary rows.
Reports include primary raw-scale macro R2 and secondary authors_joint_r2.

## Existing setup, no jobs started

Keep the prepared cohort, folds, configuration, model resources and baseline
results. Pull only while pipeline jobs are neither queued nor running. Restore
the Python/CUDA modules for the existing environments, then use:

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

Setup verifies/reuses existing resources and environments. The encoder pilot
measures the first HVG fold. After it finishes, refresh the actual project quota
and run `encoders` to prepare the remaining folds. The `pilot` command then
exercises all pending input/model pairs on all seventeen targets in one saved
fold. After it succeeds and the remaining quota is updated, `submit --execute`
creates the single production array. See the unified guide for exact commands.

## Preserving previous results

The baseline snapshot protects prepared data, split files, original TabPFN
resources and all 60 completed simple folds. Additional model resources are
allowed. Old pilot results remain in their original directories; the expanded
pilot uses its own timing location and cannot reuse incompatible older estimates.
Matching full-expression embeddings, fold-specific embeddings and target results
are reused. Never delete predictions or edit fingerprints to bypass provenance.

## GenePT archive filename recovery

The published Zenodo v2 archive names its model-3 payload
`GenePT_gene_protein_embedding_model_3_text.pickle.` with a trailing period,
although the record description omits it. The loader accepts both exact names,
checks the published MD5, ignores metadata and rejects ambiguous payloads. If an
older checkout failed during setup, pull the fix and repeat setup; the cached ZIP
and completed model downloads are verified and reused. A failed
`setup && embeddings && pilots` chain did not submit later jobs.

The Hugging Face unauthenticated-request warning is separate from that filename
error. Keep credentials in the cluster environment rather than tracked files.

## Failure and verification

Saved IDs, active-job guards, dependency checks and full-array accounting remain
in effect. Inspect logs before retrying; cancel a failed waiting dependency chain
rather than clearing its IDs. The report requires all production tasks to succeed
and applies the same saved splits to every method. It is not an exact historical
paper reproduction.

The Linux CI suite uses synthetic data, mocked encoders and fake Slurm commands;
it downloads no scientific data or model weights and submits no real jobs.
