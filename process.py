"""Extraction respiratoire sur la machine HABS (participants avec >= 2 séances de repos).

À NE LANCER QU'APRÈS VALIDATION DE L'ÉQUIPE.
Le temps du job, ce fichier remplace process.py (la machine HABS n'exécute que process.py) :
copier son contenu dans process.py, pousser, Actions -> HABS -> Run workflow, puis remettre
l'ancien process.py. resp_segmentation.py doit être à la racine du repo, à côté de process.py.

Résultats (dans out/) : cycles_tous.csv, signatures_seances.csv, qualite_seances.csv,
echecs.csv, parametres.csv. Aucune figure n'est produite.
"""
from pathlib import Path

from resp_segmentation import extraire_tout


def process(data_dir: Path, out_dir: Path) -> None:
    extraire_tout(data_dir, out_dir, workers=16)
