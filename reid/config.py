"""Shared constants: paths, seed, dataset layout, signal-processing choices."""

from pathlib import Path

SEED = 1337

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
CACHE = REPO / "cache"
FEATURES = CACHE / "features"

HF_REPO = "HABSteam/multimodal-eeg-emotion"

#: Task name in file names -> short flag used in tables (has_rest, ...).
TASKS = {"resting-state": "rest", "artefact": "artefact", "emotion-video": "video"}

#: Modalities used by the project. Oculometry (gaze, pupil, eyeball centres) is
#: excluded on purpose: its stream is not reliably synchronised with the others,
#: and the eyeball-centre distance is a morphological shortcut, not physiology.
MODALITIES = ("eeg", "ecg", "eda", "resp")
RECORD_TYPES = {"EEG": "eeg", "ECG": "ecg", "GSR": "eda", "RESPIRATION": "resp"}

#: Analysis windows (s). Every task is cut into windows of this length inside its
#: sub-segments; artefact actions keep their own 5 s length.
WINDOW_S = 10.0
EPOCH_S = 2.0  # EEG epochs inside a window, 50 % overlap
PAD_S = 2.0  # read this much signal around each window to absorb filter edges

#: EEG preprocessing: band-pass below the 50 Hz mains line, average reference.
EEG_BAND = (1.0, 45.0)
PSD_FMAX = 45.0
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
#: Covariance filter bank (broadband first).
COV_BANDS = {
    "broad": EEG_BAND,
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
REJECT_MAD = 5.0  # epoch rejected if peak-to-peak > median + 5 MAD (per session)


def default_data_dir() -> Path:
    """Dataset root: /data on the GPU platform, ./data locally, else ./sample."""
    for candidate in (Path("/data"), REPO / "data", REPO / "sample"):
        if (candidate / "CURATED").is_dir():
            return candidate
    return REPO / "sample"
