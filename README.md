# Killifish Behavior and Heart-Rate Analysis

This repository contains analysis scripts and notebooks for killifish behavior, DeepLabCut pose-derived movement metrics, bout diagnostics, and video-based heart-rate estimation.

Large raw data, generated results, trained DeepLabCut project state, videos, and local spreadsheets are intentionally excluded from Git. Path defaults are centralized in `kilifish_paths.py` and can be overridden with environment variables for another machine or storage layout.

## Repository Contents

- `kilifish_paths.py` - shared path defaults and environment-variable overrides.
- `scripts/` - behavior, heart-rate, overlay, and preprocessing scripts.
- `notebooks/archive/deeplabcut/` - DLC setup/tracking provenance notebooks only.
- `scripts/heart_rate/extract_heart_rate.py` - batch heart-rate extraction from tracked or static ROIs.
- `scripts/overlays/render_bout_overlay.py` - bout overlay and robust/simple detector comparison utility.
- `docs/HR_README.md` - heart-rate workflow notes.

## Data and Results Layout

Generated outputs are expected under:

```text
results/
  behavior/
  heart_rate/
```

Raw and processed video/data folders are local-only and ignored by Git:

```text
data/
  raw/
    archive/
    killifish-v2/
    yoshidak/
      behavior/
      heart_rate/
  processed/
    killifish-v2-encoded/
  metadata/
DeepLabCut/FishPose*/
```

DeepLabCut project folders remain under `DeepLabCut/FishPose*/` for now. Move those only after updating or regenerating DLC project `config.yaml` files that contain absolute paths.

Common path overrides:

```text
KILIFISH_ROOT
KILIFISH_DATA_ROOT
KILIFISH_RAW_DATA_ROOT
KILIFISH_PROCESSED_DATA_ROOT
KILIFISH_RESULTS_ROOT
KILIFISH_DLC_DIR
KILIFISH_ARCHIVE_ROOT
KILIFISH_V2_ROOT
KILIFISH_V2_ENCODED_ROOT
KILIFISH_YOSHIDAK_ROOT
KILIFISH_YOSHIDAK_BEHAVIOR_ROOT
KILIFISH_YOSHIDAK_HEART_RATE_ROOT
```

## Environment

The project has two environment styles:

1. Lightweight script dependencies from `requirements.txt`.
2. The full DeepLabCut conda environment in `envs/deeplabcut.yml`.

For basic analysis scripts:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For DeepLabCut workflows, use the conda environment described in `envs/deeplabcut.yml`.

## Common Commands

Run the YOSHIDAK behavior pipeline:

```bash
python scripts/behavior/yoshidak_behavior_pipeline.py
```

Run AM/PM effects on the current YOSHIDAK output:

```bash
python scripts/behavior/analyze_yoshidak_am_pm_effects.py
```

Run the 2.0 to 2.5 month tank-age analysis:

```bash
python scripts/behavior/analyze_tank_age_2_to_2p5.py
```

Generate tracked heart-rate features:

```bash
python scripts/heart_rate/run_heart_rate_pipeline.py
```

Generate a bout overlay for one video:

```bash
python scripts/overlays/render_bout_overlay.py --input video.mp4 --dlc pose.csv --outdir overlays --compare_simple --simple_mode v3
```

If the data lives somewhere else, set `KILIFISH_ROOT` or one of the path variables listed above.

## GitHub Notes

Before pushing publicly, choose and add a license. No license is included yet because that decision should match the intended sharing terms for the data and code.
