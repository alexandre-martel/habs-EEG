"""Feature extraction on the bundled P001 sample (skipped if absent)."""

import json

import numpy as np
import pytest

from reid.config import REPO
from reid.features import run_extraction
from reid.segments import windows

SAMPLE = REPO / "sample"
pytestmark = pytest.mark.skipif(not (SAMPLE / "CURATED").is_dir(), reason="no sample/")


def test_windows_follow_markers() -> None:
    meta = json.loads((SAMPLE / "CURATED/P001/P001_day-1_metadata.json").read_text())
    rest = windows(meta, "resting-state")
    assert [w.condition for w in rest] == ["rest_eo"] * 3 + ["rest_ec"] * 6
    artefact = [w.condition for w in windows(meta, "artefact")]
    assert artefact.count("art_eo") == 12 and "art_jaw" in artefact


def test_rest_extraction_produces_documented_keys(tmp_path) -> None:
    counts = run_extraction(SAMPLE, tmp_path, "rest")
    assert counts == {"done": 1, "skipped": 0, "failed": 0}
    feats = np.load(tmp_path / "rest" / "P001_day-1.npz")
    n = len(feats["w_condition"])
    assert feats["w_eeg_psd"].shape == (n, 62, 89)
    assert feats["w_eeg_cov"].shape == (n, 4, 62, 62)
    assert np.isfinite(feats["w_eeg_psd"]).all()
    for key in ("w_ecg", "w_ecg_beat", "w_eda", "w_resp"):
        assert len(feats[key]) == n
    assert 40 < np.nanmedian(feats["w_ecg"][:, 0]) < 140  # heart rate, bpm
    # second run: resumable, nothing recomputed
    assert run_extraction(SAMPLE, tmp_path, "rest")["skipped"] == 1
