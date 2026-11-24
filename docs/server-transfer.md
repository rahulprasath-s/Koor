# Transfer source from the GPU server

The server project is `~/Koor-building-reconstruction-v1`. The exporter reads selected files from that directory and produces an archive. It does not stop training, modify the project's source, request passwords, or read SSH/Hugging Face credential stores.

## Create the archive

Download the exporter from this repository and execute it in the server terminal already connected through VS Code:

```bash
curl -fL https://raw.githubusercontent.com/rahulprasath-s/Koor/main/tools/export_gpu_sources.py \
  -o /tmp/export_koor_sources.py
python3 /tmp/export_koor_sources.py --output ~/Koor-source-export.zip
```

The standard-library script needs no training environment or GPU. It includes project `scripts/*.py`, selected Point2Roof model/CUDA source and configuration, optional top-level configuration/documentation, and small summary metadata. Existing ZIP destinations are not overwritten. Its manifest records paths and SHA-256 hashes. It validates Python syntax and stops if it detects a private-key block or a recognizable literal access token.

It excludes raw datasets, model weights, prediction dumps, virtual environments, compiled extensions, build products, `.git` directories, `.env` files and credentials.

## Transfer and import

In VS Code's Remote SSH Explorer, locate `~/Koor-source-export.zip`, right-click it and choose **Download**. Attach that ZIP to the code-review chat so the source can be imported into the local checkout, reviewed and pushed to GitHub.

Another option is `scp` from a machine whose SSH login already works. The exporter does not set up SSH authentication; a VS Code password-authenticated session does not automatically give a separate local process access to the server.

Before import, inspect the archive manifest and source changes. Reconcile the current server files with the released v1 snapshot and preserve the recorded benchmark results. Verify the imported Python source syntax and review the staged Git file list before committing. The archive itself is not committed.

## V2 source from Alex

The v2 source archive was exported from Alex after the four paired held-out evaluations. It contains only five Python scripts, four batch scripts, aggregate reports, training history, split IDs and provenance. The archive manifest hashes were verified on import; shared v1 source hashes matched this repository.

The maintained GitHub layout puts Python under `v2/scripts/` and Slurm jobs under `v2/slurm/`. The exact original flat files are kept in `v2/provenance/alex_source/`, with their [export manifest](../v2/provenance/export_manifest.json). They record the source used by the archived checkpoint, not a separate runnable project.

Runtime data stays at the repository root under ignored `data/`, `models/`, `runs/` and `evaluations/`. A repository cloned into `~/Koor-building-reconstruction-v2` still contains both `v1/` and `v2/` source trees; v2 shares the model architecture and CUDA source in `v1/Point2Roof`.

Do not replace the original Alex source in the middle of a run. Its resume checks include the original source hashes and absolute prepared-data lists. The maintained scripts deliberately use a new full-run directory, `runs/koor-v2-mixed-distill`, so a fresh run cannot silently overwrite or resume the archived `runs/koor-mixed-distill-alex-v2` experiment. Keep the old workspace for that checkpoint's resume/provenance; no server files were removed by the GitHub import.

Never push the whole runtime folder. The ignore rules exclude models, checkpoints, raw datasets, logs, environments and prediction dumps. Copy only reviewed aggregate results into the versioned `v2/evaluation/` directory. Model weights are released separately through Hugging Face.
