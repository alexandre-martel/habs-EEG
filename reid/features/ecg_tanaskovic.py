"""ECG fiducial features of Tanasković et al. (2025), "Interpretable machine
learning for cardiogram-based biometrics" (arXiv 2510.19775), ECG part only.

Re-implemented in Python from the paper and from the feature-extraction
script of the authors' earlier study (CardioPRINT,
github.com/Luck032/CardioPRINT-based-biometric-identification-with-machine-learning,
CardioPrint_feature_extraction_two_segments.R, commit a2b4bfc). Durations
that the R script gives in samples at 2000 Hz are converted to milliseconds
so that the code works at any sampling rate (HABS: 1000 Hz).

Pipeline, as in the R script:

1. 4th-order Butterworth band-pass 1-40 Hz, zero phase (filtfilt).
2. Pan-Tompkins variant: first difference, squared, trailing 150 ms moving
   sum; threshold = 0.3 x the mean of the maxima of 10 evenly spaced 2.5 s
   chunks; an R peak is the first sample above threshold, then 200 ms
   blanking; each peak is moved to the maximum of the filtered ECG within
   +-30 ms.
3. Templates: mean of 10 consecutive beats, -250/+500 ms around R; the RR
   of a template is the mean of the 10 RR intervals that follow its beats.
4. Fiducial points on each template: Q and S (first local minima left and
   right of R, searched from 10 ms), T peak (maximum from S + 50 ms to
   R + 300 ms), T start and T end (trapezium-area method around the
   inflection points of the T wave).
5. 16 features (below). Intervals except QRS are Bazett-corrected
   (divided by sqrt(RR in s)), as in the R script.

Not in the R script, choices of this module (to be checked against the 2025
paper if its code is released):

* the four slopes, defined as amplitude difference / time difference
  (per ms) between the two fiducial points they name;
* ST_int (computed in the R script but not exported): (T end - S) / sqrt(RR);
* the templates are consecutive, non-overlapping groups of 10 beats inside
  each segment (the R script takes 4 groups at fixed positions; the 2025
  paper reports 9 per 2-min segment without saying where).

* T polarity: the R script assumes an upright T wave (lead II). On the HABS
  infraclavicular lead the T wave can be inverted, so T, T start and T end
  are searched on the template multiplied by the T polarity (+1/-1, chosen
  per template from the larger deviation from the template median, taken as
  the isoelectric level, between S + 50 ms and R + 300 ms). Amplitudes and slopes stay signed, on the original signal.

Kept as in the R script although it looks unintended: ECGQRScrest divides
T(peak) - Q, not R - Q, by the RMS of the QRS complex.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

NAMES = (
    "QRS_int", "T_int", "QT_int", "ST_int",
    "RS_amp", "RQ_amp", "RT_amp", "TT1_amp", "TT2_amp",
    "QR_slope", "RS_slope", "TT1_slope", "TT2_slope",
    "ECGQRScrest", "ECGTcrest", "RR",
)
FIDUCIALS = ("Q", "R", "S", "Tstart", "T", "Tend")
PRE_S, POST_S = 0.25, 0.50  # template window around R
N_AVG = 10  # beats per template


def ms(t: float, fs: float) -> int:
    return int(round(t * fs / 1000))


def bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    sos = butter(4, [1, 40], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def pan_tompkins(xf: np.ndarray, fs: float) -> np.ndarray:
    """R peaks of a band-passed ECG (sample indices), R-script variant."""
    n = len(xf)
    quad = np.diff(xf, append=xf[-1]) ** 2
    win = ms(150, fs)
    csum = np.concatenate([[0.0], np.cumsum(quad)])
    mov = np.zeros(n)
    mov[win - 1 :] = csum[win:] - csum[: n - win + 1]  # trailing sum of `win` samples
    chunk = ms(2500, fs)
    starts = [round(j * n / 10) for j in range(10)]
    threshold = 0.3 * np.mean([mov[s : s + chunk].max() for s in starts])
    blank, half = ms(200, fs), ms(30, fs)
    peaks, i = [], blank
    while i < n:
        if mov[i] > threshold:
            peaks.append(i)
            i += blank
        else:
            i += 1
    out = []
    for p in peaks:
        lo, hi = p - half, p + half + 1
        if lo >= 0 and hi <= n:
            out.append(lo + int(np.argmax(xf[lo:hi])))
    return np.asarray(out, dtype=int)


def templates(xf, peaks, fs, start, stop):
    """Consecutive 10-beat templates inside [start, stop) (samples).
    Returns [(template, rr_s, first_peak_index)]."""
    pre, post = ms(PRE_S * 1000, fs), ms(POST_S * 1000, fs)
    p = peaks[(peaks - pre >= max(start, 0)) & (peaks + post <= min(stop, len(xf)))]
    out = []
    for k in range(0, len(p) - N_AVG, N_AVG):  # needs beat k+10 for the 10th RR
        beats = p[k : k + N_AVG]
        rr = np.diff(p[k : k + N_AVG + 1]) / fs
        tpl = np.mean([xf[b - pre : b + post] for b in beats], axis=0)
        out.append((tpl, float(rr.mean()), int(beats[0])))
    return out


def fiducials(t: np.ndarray, fs: float) -> dict | None:
    """Q, R, S, Tstart, T, Tend (indices in the template), or None."""
    n = len(t)
    r = ms(PRE_S * 1000, fs)
    d = np.diff(t)  # d[k] = t[k+1] - t[k]
    d2 = np.diff(d)
    try:
        i = ms(10, fs)
        while d[r - i] > 0:
            i += 1
        q = r - i
        j = ms(10, fs)
        while d[r + j] < 0:
            j += 1
        s = r + j
        lo, hi = s + ms(50, fs), n - ms(200, fs)
        if hi <= lo:
            return None
        # T polarity (not in the R script, which assumes an upright T as in
        # lead II): the HABS infraclavicular lead often has an inverted T. The
        # T-wave rules below run on y = polarity * t, so on an upright wave.
        base = np.median(t)  # ~isoelectric level (Q can be deep on this lead)
        polarity = 1 if t[lo:hi].max() - base >= base - t[lo:hi].min() else -1
        y = polarity * t
        dy2 = polarity * d2
        tp = lo + int(np.argmax(y[lo:hi]))

        off = max(ms(5, fs), 1)
        i = 1
        while dy2[tp + i + off] < 0:
            i += 1
        a_right = tp + i + off
        b_right = min(a_right + ms(100, fs), n - 1)
        i = 1
        while dy2[tp - i - off] < 0:
            i += 1
        a_left = tp - i - off
        b_left = s + ms(50, fs)

        k = np.arange(1, b_right - a_right)
        if not len(k):
            return None
        area = 0.5 * (y[a_right] - y[a_right + k]) * (2 * b_right - 2 * a_right - k)
        t_end = a_right + int(k[np.argmax(area)])

        k = np.arange(1, a_left - b_left)
        if not len(k):
            return None
        area = np.abs(0.5 * (y[a_left] - y[a_left - k]) * (2 * b_left - 2 * a_left + k))
        t_start = a_left - int(k[np.argmax(area)])
    except IndexError:
        return None
    if not (q < r < s < t_start < tp < t_end):
        return None
    return {"Q": q, "R": r, "S": s, "Tstart": t_start, "T": tp, "Tend": t_end,
            "T_polarity": polarity}


def _crest(seg: np.ndarray, peak: float) -> float:
    rms = np.sqrt(np.mean((seg - seg[0]) ** 2))
    return (peak - seg[0]) / rms if rms > 0 else np.nan


def template_features(t: np.ndarray, f: dict, rr_s: float, fs: float) -> dict:
    """The 16 features of one template (intervals in ms, amplitudes in the
    signal's unit, slopes in unit/ms, RR in ms)."""
    to_ms = 1000 / fs
    bz = np.sqrt(rr_s)
    q, r, s, ts, tp, te = (f[k] for k in FIDUCIALS)
    v = t
    return {
        "QRS_int": (s - q) * to_ms,
        "T_int": (te - ts) * to_ms / bz,
        "QT_int": (te - q) * to_ms / bz,
        "ST_int": (te - s) * to_ms / bz,
        "RS_amp": v[r] - v[s],
        "RQ_amp": v[r] - v[q],
        "RT_amp": v[r] - v[tp],
        "TT1_amp": v[tp] - v[ts],
        "TT2_amp": v[tp] - v[te],
        "QR_slope": (v[r] - v[q]) / ((r - q) * to_ms),
        "RS_slope": (v[s] - v[r]) / ((s - r) * to_ms),
        "TT1_slope": (v[tp] - v[ts]) / ((tp - ts) * to_ms),
        "TT2_slope": (v[te] - v[tp]) / ((te - tp) * to_ms),
        "ECGQRScrest": _crest(v[q : s + 1], v[tp]),
        "ECGTcrest": _crest(v[ts : te + 1], v[tp]),
        "RR": 1000 * rr_s,
    }


BEAT_POINTS = 75  # template waveform resampled to one point every 10 ms
BEAT_NAMES = tuple(f"beat_{i:02d}" for i in range(BEAT_POINTS))


def beat_columns(tpl: np.ndarray) -> dict:
    """The whole template (-250/+500 ms around R), resampled to BEAT_POINTS
    points, in the stored ECG units: the full beat shape, for comparison with
    the 16 fiducial features."""
    v = np.interp(np.linspace(0, len(tpl) - 1, BEAT_POINTS), np.arange(len(tpl)), tpl)
    return dict(zip(BEAT_NAMES, v.tolist()))


def extract(x: np.ndarray, fs: float, segments: list[tuple[str, int, int]]):
    """Filter and detect on the whole recording, then one row per template of
    each (name, start, stop) segment. Returns (rows, details) where details
    keep the templates and fiducials for plotting."""
    xf = bandpass(x, fs)
    peaks = pan_tompkins(xf, fs)
    rows, details = [], []
    for name, start, stop in segments:
        for k, (tpl, rr, first) in enumerate(templates(xf, peaks, fs, start, stop)):
            f = fiducials(tpl, fs)
            row = {"segment": name, "template": k, "t_first_beat_s": first / fs,
                   "fiducials_ok": f is not None,
                   "T_polarity": f["T_polarity"] if f else np.nan}
            row |= template_features(tpl, f, rr, fs) if f else dict.fromkeys(NAMES, np.nan)
            row |= beat_columns(tpl)
            rows.append(row)
            details.append((name, k, tpl, f))
    return rows, details, xf, peaks
