# Pipeline d'évaluation commun : mode d'emploi

Tous les modèles de l'équipe (appariement, classifieurs, world models) sont
évalués par le même code, avec **les mêmes plis** (`results/splits/folds.json`,
commité une fois pour toutes).

## La question posée

Pour chaque personne de test, on dispose d'une session de **galerie** (jour 1)
et d'une session **probe** (jour 2). Le modèle produit une matrice de scores
`S[n_probe, n_galerie]` (plus grand = plus probablement la même personne). On
regarde si la bonne session de galerie arrive en tête.

## L'interface à implémenter

```python
class ReIDModel(Protocol):
    name: str

    def fit(self, train: SessionSet) -> None: ...
    def score(self, gallery: SessionSet, probe: SessionSet) -> np.ndarray: ...
```

- `SessionSet.sessions` est un DataFrame : `participant_id, day, task, cohort,
  age, gender, part`.
- `for feats in sessions:` (ou `sessions.load(i)`) renvoie les features en cache
  d'une session, sous forme de dict de tableaux. Les clés `w_*` ont une ligne par
  fenêtre de 10 s (`w_condition` : `rest_ec`, `rest_eo`, `art_eo`, `art_blink`,
  `video`… ; `w_eeg_psd`, `w_eeg_cov`, `w_ecg`, `w_eda`, `w_resp`). Les autres
  clés sont partagées (`freqs`, `ch_names`, `cov_bands`…). Le détail est dans
  `scripts/reid_extract.py`.
- `fit` ne voit **jamais** une session d'une personne de test : ni sa probe, ni
  sa galerie. L'enrôlement (apprendre la galerie) se fait dans `score`. Le
  runner le vérifie par une assertion.

### Modèle d'embeddings (world model, LDA, réseau siamois…)

```python
from reid.eval.interface import EmbeddingModel


class MyEmbedding(EmbeddingModel):
    name = "jepa_v1"

    def fit(self, train): ...  # optionnel
    def embed(self, sessions):  # [n_sessions, d]
        return np.stack([my_encoder(f) for f in sessions])
```

`score` = similarité cosinus, fournie par la classe de base.

### Classifieur (une classe par personne)

Le classifieur est entraîné sur les fenêtres de la galerie (une classe par
session de galerie), puis ses probabilités sont moyennées sur les fenêtres de
chaque probe.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from reid.eval.interface import ClassifierModel


def band_power(feats):  # [n_fenêtres, 62 * 89]
    return np.log10(feats["w_eeg_psd"]).reshape(len(feats["w_eeg_psd"]), -1)


model = lambda: ClassifierModel(
    "logreg_psd",
    lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    band_power,
)
```

Si le classifieur a besoin d'un pré-entraînement sur les personnes
d'entraînement (extracteur partagé), passe `pretrain=fonction(train)`.

## Lancer une évaluation

```python
from reid.eval.protocol import FeatureStore, load_folds
from reid.eval.runner import evaluate
from reid.inventory import load_inventory

inv = load_inventory("./data")  # joint cohorte/âge/genre depuis les tables
run = evaluate(model, inv, load_folds(), "d1_to_d2", task="rest", store=FeatureStore())
```

`model` est une **fabrique** (fonction sans argument, ou la classe) : un modèle
neuf est créé pour chaque pli. Pour l'enregistrer dans les scripts, ajoute-le à
`reid/baselines/__init__.py:MODELS`, puis :

```bash
python scripts/reid_run.py --model mon_modele --scenario all
python scripts/reid_report.py      # leaderboard.csv + figures
```

## Scénarios

| nom | galerie → probe | remarque |
|---|---|---|
| `d1_to_d2` | jour 1 → jour 2 | principal |
| `d1_to_d3`, `d2_to_d3` | cohorte 3 | effet du temps, ~25 personnes |
| `within_cohort` | jour 1 → jour 2 | appariement limité à la même cohorte |
| `with_distractors` | jour 1 + personnes à un seul jour → jour 2 | galerie élargie |
| `same_day_ceiling` | 1re moitié du jour 1 → 2e moitié | plafond optimiste, jamais un résultat |

`--task` choisit la tâche (`rest`, `artefact`, `video`). `--probe-task` permet le
transfert entre tâches (galerie = repos, probe = vidéo).

## Ce qui est rapporté

Top-1, top-5, MRR, rang médian, précision après **appariement hongrois**
(one-to-one), AUC et **EER** en vérification, et le **hasard** (1/N). Les
intervalles de confiance à 95 % viennent d'un bootstrap sur les personnes
(1 000 tirages). On donne la valeur par pli et la valeur poolée. Les ex-æquo
sont départagés au hasard, en espérance : un score constant donne exactement le
hasard.

Contrôles automatiques : labels de galerie mélangés (doit donner le hasard),
démographie seule (âge + genre : **tout modèle doit faire nettement mieux**),
prédictibilité de la cohorte à partir des embeddings. Chaque run est écrit dans
`results/runs/*.json` : métriques agrégées uniquement, aucune donnée par
personne.

## Limites à connaître

- Avec 5 plis, une galerie compte environ 43 personnes : hasard ≈ 2,3 %. La
  courbe de taille de galerie s'arrête donc à la taille d'un pli. Pour aller
  plus loin, il faut moins de plis (`--n-folds 2` → environ 106 personnes).
- Le bloc artefact n'existe qu'un jour par personne : pas de scénario
  artefact → artefact entre jours.
