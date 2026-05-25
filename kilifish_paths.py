from __future__ import annotations

import os
from pathlib import Path


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve()


PROJECT_ROOT = env_path("KILIFISH_ROOT", Path(__file__).resolve().parent)
DATA_ROOT = env_path("KILIFISH_DATA_ROOT", PROJECT_ROOT / "data")
RAW_DATA_ROOT = env_path("KILIFISH_RAW_DATA_ROOT", DATA_ROOT / "raw")
PROCESSED_DATA_ROOT = env_path("KILIFISH_PROCESSED_DATA_ROOT", DATA_ROOT / "processed")
RESULTS_ROOT = env_path("KILIFISH_RESULTS_ROOT", PROJECT_ROOT / "results")

DEEPLABCUT_DIR = env_path("KILIFISH_DLC_DIR", PROJECT_ROOT / "DeepLabCut")
ARCHIVE_ROOT = env_path("KILIFISH_ARCHIVE_ROOT", RAW_DATA_ROOT / "archive")
KILLIFISH_V2_ROOT = env_path("KILIFISH_V2_ROOT", RAW_DATA_ROOT / "killifish-v2")
KILLIFISH_V2_ENCODED_ROOT = env_path(
    "KILIFISH_V2_ENCODED_ROOT",
    PROCESSED_DATA_ROOT / "killifish-v2-encoded",
)

YOSHIDAK_ROOT = env_path("KILIFISH_YOSHIDAK_ROOT", RAW_DATA_ROOT / "yoshidak")
YOSHIDAK_BEHAVIOR_ROOT = env_path(
    "KILIFISH_YOSHIDAK_BEHAVIOR_ROOT",
    YOSHIDAK_ROOT / "behavior",
)
YOSHIDAK_HEART_RATE_ROOT = env_path(
    "KILIFISH_YOSHIDAK_HEART_RATE_ROOT",
    YOSHIDAK_ROOT / "heart_rate",
)

OUT_V4_ROBUST = env_path("KILIFISH_OUT_V4_ROBUST", RESULTS_ROOT / "behavior" / "v4_robust")
OUT_V5_STORY = env_path("KILIFISH_OUT_V5_STORY", RESULTS_ROOT / "behavior" / "v5_story")
OUT_V5_WEIGHT = env_path("KILIFISH_OUT_V5_WEIGHT", RESULTS_ROOT / "behavior" / "v5_weight")
OUT_V6 = env_path("KILIFISH_OUT_V6", RESULTS_ROOT / "behavior" / "v6")
OUT_V6_OLD_15FPS = env_path(
    "KILIFISH_OUT_V6_OLD_15FPS",
    RESULTS_ROOT / "behavior" / "v6_old_15fps",
)
OUT_V6_YOSHIDAK = env_path(
    "KILIFISH_OUT_V6_YOSHIDAK",
    RESULTS_ROOT / "behavior" / "v6_yoshidak",
)
OUT_BOUT_DIAGNOSTICS = env_path(
    "KILIFISH_OUT_BOUT_DIAGNOSTICS",
    RESULTS_ROOT / "behavior" / "bout_diagnostics_compare",
)
OUT_BOUT_PATH_INVARIANCE = env_path(
    "KILIFISH_OUT_BOUT_PATH_INVARIANCE",
    RESULTS_ROOT / "behavior" / "bout_path_invariance",
)
OUT_V6_HR = env_path("KILIFISH_OUT_V6_HR", RESULTS_ROOT / "heart_rate" / "v6_hr")
HR_RESULTS = env_path("KILIFISH_HR_RESULTS", RESULTS_ROOT / "heart_rate" / "hr_results")
HR_RESULTS_BATCH = env_path(
    "KILIFISH_HR_RESULTS_BATCH",
    RESULTS_ROOT / "heart_rate" / "hr_results_batch",
)
