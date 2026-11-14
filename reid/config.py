"""Shared constants: paths, seed, dataset layout."""

from pathlib import Path

SEED = 1337

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
CACHE = REPO / "cache"

HF_REPO = "HABSteam/multimodal-eeg-emotion"

#: Task name in file names -> short flag used in tables (has_rest, ...).
TASKS = {"resting-state": "rest", "artefact": "artefact", "emotion-video": "video"}


def default_data_dir() -> Path:
    """Dataset root: /data on the GPU platform, ./data locally, else ./sample."""
    for candidate in (Path("/data"), REPO / "data", REPO / "sample"):
        if (candidate / "CURATED").is_dir():
            return candidate
    return REPO / "sample"
