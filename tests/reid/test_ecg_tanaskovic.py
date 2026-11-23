"""reid.features.ecg_tanaskovic on simulated ECG (upright and inverted T)."""

import neurokit2 as nk
import numpy as np

from reid.features import ecg_tanaskovic as tk

FS = 1000


def _ecg(seed=0):
    return nk.ecg_simulate(duration=40, sampling_rate=FS, heart_rate=70,
                           noise=0.01, random_state=seed)


def test_r_peaks_match_heart_rate():
    xf = tk.bandpass(_ecg(), FS)
    peaks = tk.pan_tompkins(xf, FS)
    rr = np.diff(peaks) / FS
    assert abs(60 / np.median(rr) - 70) < 3


def test_features_physiological_on_upright_t():
    x = _ecg()
    rows, details, _, _ = tk.extract(x, FS, [("rest", 0, len(x))])
    ok = [r for r in rows if r["fiducials_ok"]]
    assert len(rows) >= 3 and len(ok) == len(rows)
    assert all(r["T_polarity"] == 1 for r in ok)
    for r in ok:
        assert 20 < r["QRS_int"] < 150
        assert 250 < r["QT_int"] < 550
        assert r["RQ_amp"] > 0 and r["RS_amp"] > 0
        assert abs(r["RR"] - 60000 / 70) < 60
        assert set(tk.NAMES) <= set(r)


def test_inverted_t_is_detected():
    """Flip the T wave only (R kept upright): polarity -1, ordered fiducials."""
    x = _ecg(1)
    xf = tk.bandpass(x, FS)
    peaks = tk.pan_tompkins(xf, FS)
    y = x.copy()
    base = np.median(x)  # isoelectric level of the simulated signal
    for p in peaks:  # mirror R+120..R+400 ms (the T wave) around the baseline
        seg = slice(p + 120, min(p + 400, len(y)))
        y[seg] = 2 * base - y[seg]
    rows, details, _, _ = tk.extract(y, FS, [("rest", 0, len(y))])
    ok = [d for d in details if d[3]]
    assert ok, "no template with fiducials"
    assert all(f["T_polarity"] == -1 for *_, f in ok)
    for *_, f in ok:
        assert f["Q"] < f["R"] < f["S"] < f["Tstart"] < f["T"] < f["Tend"]
