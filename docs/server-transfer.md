# Transfer source from the GPU server

The server holds `~/Koor-building-reconstruction-v1` and `~/Koor-building-reconstruction-v2`. The exporter reads selected files from those directories and produces an archive. It does not stop training, modify either project's source, request passwords, or read SSH/Hugging Face credential stores.

## Create the archive

Download the exporter from this repository and execute it in the server terminal already connected through VS Code:

```bash
curl -fL https://raw.githubusercontent.com/rahulprasath-s/Koor/main/tools/export_gpu_sources.py \
  -o /tmp/export_koor_sources.py
python3 /tmp/export_koor_sources.py --output ~/Koor-source-export.zip
```

The standard-library script needs no training environment or GPU. It includes project `scripts/*.py`, selected Point2Roof model/CUDA source and configuration, optional top-level configuration/documentation, and small summary metadata. Existing ZIP destinations are not overwritten. Its manifest records paths and SHA-256 hashes. It validates Python syntax and stops if it detects a private-key block or a recognizable literal access token.

It excludes raw datasets, model weights, prediction dumps, virtual environments, compiled extensions, build products, `.git` directories, `.env` files and credentials. Building3D contributes only a summary of its preparation manifest, without per-building geometry or source coordinates.

## Transfer and import

In VS Code's Remote SSH Explorer, locate `~/Koor-source-export.zip`, right-click it and choose **Download**. Attach that ZIP to the code-review chat so the source can be imported into the local checkout, reviewed and pushed to GitHub.

Another option is `scp` from a machine whose SSH login already works. The exporter does not set up SSH authentication; a VS Code password-authenticated session does not automatically give a separate local process access to the server.

Before import, inspect the archive manifest and source changes. Reconcile the current server files with the released v1 snapshot; do not replace benchmark results or claim completed v2 training without evidence. Verify the imported Python source syntax and review the staged Git file list before committing. The archive itself is not committed.
