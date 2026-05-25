# Scripts

Canonical command-line entry points live in these folders. Superseded one-off scripts have been removed.

## Behavior

- `behavior/yoshidak_behavior_pipeline.py` - current YOSHIDAK behavior metrics pipeline.
- `behavior/encoded_behavior_pipeline_15fps.py` - encoded legacy video behavior metrics pipeline.
- `behavior/compare_bout_diagnostics.py` - compare bout diagnostics across pipelines.
- `behavior/analyze_bout_path_invariance.py` - bout path invariance analysis.
- `behavior/analyze_tank_age_2_to_2p5.py` - 2.0 to 2.5 month tank-age analysis.
- `behavior/analyze_yoshidak_am_pm_effects.py` - YOSHIDAK AM/PM effects analysis.

## Heart Rate

- `heart_rate/extract_heart_rate.py` - tracked/static ROI HR extraction.
- `heart_rate/run_heart_rate_pipeline.py` - derive HR features and merge them into downstream analyses.
- `heart_rate/export_first_frames.py` - export first frames for ROI selection.

## Overlays and Preprocessing

- `overlays/render_bout_overlay.py` - render robust/simple bout overlays from DLC tracks.
- `preprocessing/encode_videos.sh` - encode raw videos into the processed data tree.
