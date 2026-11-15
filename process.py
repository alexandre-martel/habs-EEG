"""Étape 1 du projet : découper chaque enregistrement en fenêtres et en tirer des features.

Question de recherche : peut-on retrouver une personne du jour 1 dans les
données du jour 2 ? Ce script ne fait AUCUN apprentissage. Il transforme les
BDF (des centaines de Go) en un seul petit tableau, `features.csv`, avec une
ligne par fenêtre de 15 s et des colonnes de features par modalité. Les
modèles (régression logistique, mini-modèles par signal, encodeur) s'entraînent
ensuite en local sur ce tableau.

Fenêtres : 15 s de signal, puis 5 s ignorées, et ainsi de suite (une fenêtre
toutes les 20 s), pour que deux fenêtres voisines ne se chevauchent pas.

Colonnes produites (préfixe = modalité, pour les mini-modèles par signal) :
  eeg_<électrode>_<bande>  log10 de la puissance (µV²) dans delta/theta/alpha/beta/gamma
  ecg_*                    fréquence cardiaque, variabilité (RMSSD), nombre de battements
  resp_*                   fréquence respiratoire, amplitude
  gsr_*                    niveau, variabilité et pente de la conductance de la peau
  eye_*                    diamètre des pupilles, dispersion du regard, part de données valides
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

sys.path.insert(0, str(Path(__file__).resolve().parent / "starting_notebooks"))
from bdf_loader import load_bdf  # noqa: E402

# ── Réglages ────────────────────────────────────────────────────────────────
SOURCE = "PREPROCESSED"                 # données nettoyées par HABS
# Job 1 : le repos seulement (expérience principale, ~20 Mo par fichier, rapide).
# Job 2 : remplacer par ["emotion-video"]. Les artefacts n'existent jamais sur 2 jours.
TASKS = ["resting-state"]
WINDOW_S, STEP_S = 15.0, 20.0           # 15 s de signal, une fenêtre toutes les 20 s
# Traitement par lots si 1 h ne suffit pas : participants n° FIRST à LAST-1
# dans l'ordre alphabétique (ex. 0-120, puis 120-240, puis 240-400).
FIRST, LAST = 0, 10_000
WORKERS = min(16, os.cpu_count() or 1)  # fichiers traités en parallèle

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
TO_UV = {"V": 1e6, "mV": 1e3, "uV": 1.0, "µV": 1.0, "μV": 1.0, "nV": 1e-3}


# ── Features par modalité, sur une fenêtre ─────────────────────────────────
def eeg_features(x: np.ndarray, names: list[str], fs: float) -> dict:
    """x : (électrodes, échantillons) en µV."""
    freqs, psd = welch(x, fs=fs, nperseg=int(2 * fs), axis=1)
    out = {}
    for band, (lo, hi) in BANDS.items():
        sel = (freqs >= lo) & (freqs < hi)
        power = np.trapezoid(psd[:, sel], freqs[sel], axis=1)
        for name, p in zip(names, power):
            out[f"eeg_{name}_{band}"] = np.log10(p + 1e-12)
    return out


def ecg_features(x: np.ndarray, fs: float) -> dict:
    """Détecte les battements (pics R) et en tire rythme et variabilité."""
    filtered = sosfiltfilt(butter(3, [5, 20], "bandpass", fs=fs, output="sos"), x)
    energy = filtered ** 2
    peaks, _ = find_peaks(energy, distance=int(0.33 * fs), height=np.percentile(energy, 95) * 0.3)
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.33) & (rr < 2.0)]                 # 30 à 180 bpm plausibles
    if len(rr) < 3:
        return {"ecg_hr_bpm": np.nan, "ecg_rmssd_ms": np.nan, "ecg_n_beats": len(peaks)}
    return {"ecg_hr_bpm": 60 / rr.mean(),
            "ecg_rmssd_ms": 1000 * np.sqrt(np.mean(np.diff(rr) ** 2)),
            "ecg_n_beats": len(peaks)}


def resp_features(x: np.ndarray, fs: float) -> dict:
    x = x - x.mean()
    freqs, psd = welch(x, fs=fs, nperseg=len(x))
    sel = (freqs >= 0.1) & (freqs <= 0.7)              # 6 à 42 respirations / min
    rate = freqs[sel][np.argmax(psd[sel])] * 60 if sel.any() else np.nan
    return {"resp_rate_bpm": rate, "resp_amplitude": x.std()}


def gsr_features(x: np.ndarray, fs: float) -> dict:
    t = np.arange(len(x)) / fs
    return {"gsr_mean_uS": x.mean(), "gsr_std_uS": x.std(),
            "gsr_slope_uS_per_s": np.polyfit(t, x, 1)[0]}


def eye_features(win: pd.DataFrame) -> dict:
    """Les lunettes enregistrent à ~166 Hz : les autres lignes sont vides (NaN)."""
    pupil = win[["pupildiameter_le", "pupildiameter_ri"]]
    valid = pupil.notna().all(axis=1)
    return {"eye_pupil_left_mm": pupil.iloc[:, 0].mean(),
            "eye_pupil_right_mm": pupil.iloc[:, 1].mean(),
            "eye_pupil_std_mm": pupil.stack().std(),
            "eye_gaze_x_std": win["gaze_x"].std(),
            "eye_gaze_y_std": win["gaze_y"].std(),
            "eye_valid_samples": int(valid.sum())}


# ── Un fichier → une liste de fenêtres ─────────────────────────────────────
def features_of_recording(bdf: Path) -> list[dict]:
    participant, day, task = bdf.name.split("_")[:3]
    raw, signals, channels = load_bdf(bdf)
    fs = float(raw.info["sfreq"])
    units = channels.set_index("bdf_label")["header_unit"].to_dict()
    eeg_names = channels.loc[channels.mne_type == "eeg", "bdf_label"].tolist()
    eeg = signals[eeg_names].to_numpy().T * np.array([[TO_UV[units[c]]] for c in eeg_names])

    n, width, step = len(signals), int(WINDOW_S * fs), int(STEP_S * fs)
    rows = []
    for start in range(0, n - width + 1, step):
        sl = slice(start, start + width)
        win = signals.iloc[sl]
        row = {"participant": participant, "day": day.removeprefix("day-"),
               "task": task.removeprefix("task-"), "start_s": start / fs}
        for name, compute in [
            ("eeg", lambda: eeg_features(eeg[:, sl], eeg_names, fs)),
            ("ecg", lambda: ecg_features(win["ECG"].to_numpy(), fs)),
            ("resp", lambda: resp_features(win["Resp"].to_numpy(), fs)),
            ("gsr", lambda: gsr_features(win["GSR"].to_numpy(), fs)),
            ("eye", lambda: eye_features(win)),
        ]:
            try:                                   # un capteur absent ne bloque pas les autres
                row.update(compute())
            except Exception as error:             # noqa: BLE001
                print(f"  {bdf.name} {start / fs:.0f}s: {name} ignoré ({error})")
        rows.append(row)
    del signals, eeg
    print(f"{bdf.name}: {len(rows)} fenêtres", flush=True)
    return rows


def process(data_dir: Path, out_dir: Path) -> None:
    participants = sorted(p for p in (data_dir / SOURCE).glob("P*") if p.is_dir())[FIRST:LAST]
    files = [f for p in participants for t in TASKS for f in sorted(p.glob(f"*_task-{t}_*.bdf"))]
    print(f"{len(participants)} participants, {len(files)} fichiers, {WORKERS} en parallèle")

    rows, failed = [], []
    with ProcessPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(features_of_recording, f): f for f in files}
        for future, f in futures.items():
            try:
                rows.extend(future.result())
            except Exception as error:             # noqa: BLE001
                failed.append({"file": f.name, "error": repr(error)})
                print(f"ÉCHEC {f.name}: {error}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"features_{FIRST}_{LAST}.csv", index=False)
    pd.DataFrame(failed, columns=["file", "error"]).to_csv(out_dir / f"failed_{FIRST}_{LAST}.csv", index=False)
    print(f"\n{len(table)} fenêtres × {table.shape[1]} colonnes écrites, {len(failed)} fichier(s) en échec")
