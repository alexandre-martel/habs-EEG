"""Point d'entrée du hackathon : reproduction de Demuru & Fraschini 2020.

La plateforme HABS appelle process(data_dir, out_dir) une fois :
- sur le GPU, data_dir = /data (tout le dataset, en lecture seule) ;
- en local, data_dir = le dossier passé à `habs run --data ...`.

On peut aussi le lancer directement :
    python process.py --data ./sample --out ./out
    python process.py --data ./data --out ./out --max-participants 30

Tous les réglages sont dans CONFIG, en haut de demuru_fingerprint.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demuru_fingerprint import run  # noqa: E402


def process(data_dir: Path, out_dir: Path) -> None:
    run(Path(data_dir), Path(out_dir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Demuru & Fraschini 2020 sur le dataset HABS")
    ap.add_argument("--data", required=True, type=Path, help="racine du dataset (contient PREPROCESSED/)")
    ap.add_argument("--out", required=True, type=Path, help="dossier de sortie")
    ap.add_argument("--max-participants", type=int, default=None, help="limiter pour un test rapide")
    ap.add_argument("--version", choices=["PREPROCESSED", "CURATED"], default=None)
    ap.add_argument("--jobs", type=int, default=None, help="nombre de processus (-1 = tous)")
    a = ap.parse_args()
    overrides = {}
    if a.max_participants:
        overrides["max_participants"] = a.max_participants
    if a.version:
        overrides["data_version"] = a.version
    if a.jobs:
        overrides["n_jobs"] = a.jobs
    run(a.data, a.out, overrides)
