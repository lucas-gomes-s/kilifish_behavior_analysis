# Killifish Heart-Rate from Video

The active HR utility is `scripts/heart_rate/extract_heart_rate.py`. It supports two ROI modes:

- `tracked_body_patch` uses DLC tracks to follow the fish body and is the default for current analyses.
- `static_roi` uses fixed rectangles from `config/roi_template.json` for legacy fixed-ROI workflows.

## Files

- `scripts/heart_rate/extract_heart_rate.py` - canonical batch HR extraction script.
- `scripts/heart_rate/run_heart_rate_pipeline.py` - analysis pipeline that derives HR features and merges them with behavior/survival tables.
- `scripts/heart_rate/export_first_frames.py` - helper to export video thumbnails for static ROI selection.
- `config/roi_template.json` - static ROI template when `--roi_mode static_roi` is used.
- `results/heart_rate/` - generated outputs, ignored by Git.

## Quick Start

Tracked ROI mode:

```bash
python scripts/heart_rate/extract_heart_rate.py --root /path/to/videos --outdir /path/to/results/heart_rate/hr_results --skip_overlay
```

Static ROI mode:

```bash
python scripts/heart_rate/extract_heart_rate.py --root /path/to/videos --roi_mode static_roi --roi_json config/roi_template.json --outdir /path/to/results/heart_rate/hr_results
```

Pipeline-level HR feature generation:

```bash
python scripts/heart_rate/run_heart_rate_pipeline.py
```

## Parameters

- `--bpm_min`, `--bpm_max` limit the frequency search range.
- `--channel` can be `gray`, `0` blue, `1` green, or `2` red.
- `--decimate` reduces temporal resolution to speed up processing.
- `--roi_scale` controls the tracked body patch size.
- `--track_pcutoff` and `--max_gap_frames` control DLC track quality filtering.
- `--skip_overlay` avoids rendering QC videos when only CSV features are needed.

Superseded fixed-ROI-only scripts were removed; use `--roi_mode static_roi` in the canonical extractor for that workflow.
