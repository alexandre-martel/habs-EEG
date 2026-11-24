"""Offline fusion of saved member scores (scripts/reid_offline_fusion.py)."""

import importlib.util

import numpy as np

from reid.config import REPO

spec = importlib.util.spec_from_file_location(
    "offline", REPO / "scripts/reid_offline_fusion.py"
)
off = importlib.util.module_from_spec(spec)
spec.loader.exec_module(off)


def test_seed_ensemble_beats_its_noisy_seeds(tmp_path):
    rng = np.random.default_rng(0)
    for f in range(5):
        for kind, n in (("val", 20), ("test", 30)):
            truth = rng.permutation(n)
            same = np.zeros((n, n))
            same[np.arange(n), truth] = 1
            arr = {f"S:enc_eeg_s{k}": same + rng.normal(size=(n, n)) for k in range(3)}
            arr |= {
                f"S:enc_{m}_s{k}": rng.normal(size=(n, n))
                for m in ("ecg_beat", "resp")
                for k in range(3)
            }
            arr |= {"S:lda_eeg": 0.5 * same + rng.normal(size=(n, n))}
            arr |= {"S:lda_ecg_beat": rng.normal(size=(n, n))}
            np.savez(tmp_path / f"x_{f:010x}_{kind}.npz", truth=truth, **arr)
    data = off.load([tmp_path])
    assert len(data["val"]) == len(data["test"]) == 5
    members = set(data["test"]["0000000000"]) - {"truth"}
    v = off.variants(members, [0, 1, 2])
    assert v["all5 ens3"][2] == ["enc_eeg_s0", "enc_eeg_s1", "enc_eeg_s2"]
    single = off.parts(data["test"], [["enc_eeg_s0"]], "z")
    ens = off.parts(data["test"], [v["all5 ens3"][2]], "z")
    assert off.val_metrics(ens, [1.0])[0] > off.val_metrics(single, [1.0])[0]
    Z = off.NORMS["rank"](np.array([[3.0, 1.0, 2.0]]))
    assert np.allclose(Z, [[0.5, -0.5, 0.0]])
