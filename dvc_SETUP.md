# DVC Data Versioning — Setup Notes

This project uses DVC to version the training reference dataset
(`data/processed/reference_data.csv`) and the trained model
(`pipeline/models/latest_model.json`), and to define the training step
as a reproducible pipeline stage (`dvc.yaml` / `dvc.lock`).

## IMPORTANT: dependency conflict already fixed

`dvc==3.49.0` (pinned in requirements.txt) fails on `dvc init` with:
```
ERROR: unexpected error - cannot import name '_DIR_MARK' from 'pathspec.patterns.gitwildmatch'
```
This happens because `pathspec>=1.0` removed an internal API that DVC 3.49.0
still imports. requirements.txt now also pins `pathspec==0.11.2` to avoid this.
If you ever upgrade DVC, you can likely drop that pin — but verify `dvc init`
actually works first.

## One-time setup (after cloning the repo)

```bash
# 1. Make sure dependencies are installed (includes the pathspec pin above)
pip install -r requirements.txt

# 2. DVC needs a git repo to attach to
git init   # skip if already a git repo

# 3. Initialize DVC
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# 4. Configure your real remote storage (replace with your actual S3/GCS/Azure
#    bucket or shared path -- this is NOT set to anything real by default).
#    Examples:
dvc remote add -d storage s3://your-bucket/dvc-storage
# or
dvc remote add -d storage gs://your-bucket/dvc-storage
# or, for local/network testing:
dvc remote add -d storage /path/to/shared/storage

git add .dvc/config
git commit -m "Configure DVC remote"
```

If your remote needs credentials (e.g. AWS access keys for S3), configure
those via `dvc remote modify` with `--local` so they're never committed to
git -- see https://dvc.org/doc/user-guide/data-management/remote-storage
for the exact flags per storage backend.

## Day-to-day usage

```bash
# Run the training pipeline through DVC (only re-runs if code/data actually changed)
dvc repro

# Push newly tracked data/model versions to remote storage
dvc push

# Pull data/model versions (e.g. on a fresh clone or a different machine)
dvc pull

# See what's changed without re-running anything
dvc status
```

## How dvc.yaml works here

`dvc.yaml` defines one stage, `train`, generated via:
```bash
dvc stage add -n train \
  -d pipeline/train_pipeline.py \
  -d src/feature_engineering.py \
  -d config.py \
  -o pipeline/models/latest_model.json \
  -o data/processed/reference_data.csv \
  "python pipeline/train_pipeline.py"
```

This means: if `train_pipeline.py`, `feature_engineering.py`, or `config.py`
change, `dvc repro` knows to re-run training. The outputs
(`latest_model.json`, `reference_data.csv`) are tracked by content hash in
`dvc.lock` -- DVC will skip re-running the stage if none of its declared
dependencies actually changed, even if you run `dvc repro` repeatedly.

## What's verified vs. what isn't

Tested for real, in a clean environment, with the exact pinned versions:
- `dvc init` (after the pathspec fix above)
- `dvc add` on both the reference CSV and the model file
- `dvc remote add` with a local filesystem remote
- `dvc push` — confirmed files actually land in remote storage, content-addressed by MD5
- `dvc pull` — deleted the local file, pulled it back, confirmed byte-identical
- `dvc repro` — confirmed it runs the real training pipeline, generates `dvc.lock`,
  correctly SKIPS re-running when nothing changed, and correctly re-runs when
  a dependency's content hash changes

NOT tested: a real cloud remote (S3/GCS/Azure), since no credentials were
available. The local filesystem remote test exercises the same DVC code path
for push/pull, but cloud-specific auth and networking are unverified --
test with your actual bucket before relying on this for production backups.