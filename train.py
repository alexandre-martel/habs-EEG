"""Premier modèle : retrouver une personne du jour 2 à partir de son jour 1.

Lit le tableau produit par process.py (une ligne par fenêtre de 15 s) et répond à :
« si on apprend à quoi ressemble chaque personne sur sa 1re séance, la
reconnaît-on sur sa 2e séance ? »

Usage :
    python train.py ../data/features_repos.csv

Ce que fait le script :
  1. garde les participants qui ont au moins 2 séances (jours) ;
  2. entraîne une régression logistique sur la 1re séance, teste sur la 2e ;
  3. refait la même chose pour chaque signal seul (EEG, cœur, respiration, peau, yeux) ;
  4. contrôle « même jour » : apprend et teste dans la 1re séance (score gonflé
     par l'effet de séance : l'écart avec le score jour 1 -> jour 2 mesure cet effet) ;
  5. contrôle « hasard » : mélange les identités, le score doit tomber à ~1/N.

Résultats : affichés et écrits dans resultats_train.csv.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

META = ["participant", "day", "task", "start_s"]
# Retirées : ce ne sont pas des mesures du corps mais du matériel (la cadence des
# lunettes change d'une séance à l'autre), le modèle pourrait s'en servir pour tricher.
EXCLUDED = ["eye_valid_samples"]
SIGNALS = {"EEG": "eeg_", "Cœur (ECG)": "ecg_", "Respiration": "resp_",
           "Peau (GSR)": "gsr_", "Yeux": "eye_"}
SEED = 0


def model():
    """Valeurs manquantes -> médiane ; mise à l'échelle ; régression logistique."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000))


def scores(clf, test: pd.DataFrame, cols: list[str]) -> dict:
    """Taux de bonnes réponses par fenêtre, puis par séance (les fenêtres votent)."""
    proba = pd.DataFrame(clf.predict_proba(test[cols]), columns=clf.classes_, index=test.index)
    truth = test["participant"]

    def top(p: pd.DataFrame, y: pd.Series, k: int) -> float:
        best = np.argsort(-p.to_numpy(), axis=1)[:, :k]
        return float(np.mean([y.iloc[i] in p.columns[best[i]] for i in range(len(p))]))

    session = proba.groupby([truth, test["day"]]).mean()          # moyenne des fenêtres
    session_truth = pd.Series(session.index.get_level_values(0), index=session.index)
    return {"fenetre_top1": top(proba, truth, 1), "fenetre_top5": top(proba, truth, 5),
            "seance_top1": top(session, session_truth, 1), "seance_top5": top(session, session_truth, 5)}


def split_sessions(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1re séance de chaque personne -> entraînement, 2e séance -> test."""
    rank = table.groupby("participant")["day"].rank(method="dense")
    keep = table.groupby("participant")["day"].transform("nunique") >= 2
    return table[keep & (rank == 1)], table[keep & (rank == 2)]


def main(path: str) -> None:
    table = pd.read_csv(path)
    features = [c for c in table.columns if c not in META + EXCLUDED]
    train, test = split_sessions(table)
    n = train["participant"].nunique()
    print(f"{table.participant.nunique()} participants, dont {n} avec au moins 2 séances")
    print(f"Entraînement : {len(train)} fenêtres (1re séance) | Test : {len(test)} fenêtres (2e séance)")
    print(f"Niveau du hasard : 1/{n} = {100 / n:.2f} %\n")

    rows = []

    def run(name: str, cols: list[str], tr: pd.DataFrame, te: pd.DataFrame) -> None:
        clf = model().fit(tr[cols], tr["participant"])
        rows.append({"experience": name, "n_colonnes": len(cols), **scores(clf, te, cols)})

    # 1. Expérience principale et un modèle par signal
    run("Jour 1 -> jour 2 | tous les signaux", features, train, test)
    for name, prefix in SIGNALS.items():
        run(f"Jour 1 -> jour 2 | {name} seul", [c for c in features if c.startswith(prefix)], train, test)
    run("Jour 1 -> jour 2 | tout sauf EEG", [c for c in features if not c.startswith("eeg_")], train, test)

    # 2. Contrôle « même jour » : chaque fenêtre de la 1re séance est testée à tour de rôle
    same_day = []
    for start in sorted(train["start_s"].unique()):
        held = train["start_s"] == start
        clf = model().fit(train.loc[~held, features], train.loc[~held, "participant"])
        same_day.append(scores(clf, train[held], features)["fenetre_top1"])
    rows.append({"experience": "Contrôle même jour | tous les signaux", "n_colonnes": len(features),
                 "fenetre_top1": float(np.mean(same_day))})

    # 3. Contrôle « hasard » : on mélange qui est qui dans le test
    shuffled = test.copy()
    mapping = dict(zip(sorted(test.participant.unique()),
                       np.random.default_rng(SEED).permutation(sorted(test.participant.unique()))))
    shuffled["participant"] = shuffled["participant"].map(mapping)
    run("Contrôle hasard (identités mélangées)", features, train, shuffled)

    out = pd.DataFrame(rows)
    out["hasard"] = 1 / n
    out.to_csv(Path(path).parent / "resultats_train.csv", index=False)
    pct = out.copy()
    for c in ["fenetre_top1", "fenetre_top5", "seance_top1", "seance_top5", "hasard"]:
        pct[c] = (100 * pct[c]).round(1)
    print(pct.to_string(index=False, na_rep="-"))
    print("\n(scores en % ; top1 = la bonne personne en 1er choix, top5 = dans les 5 premiers)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data/features_repos.csv")
