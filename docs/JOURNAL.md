# Journal — ré-identification jour 1 → jour 2

## Cadre éthique

On apparie uniquement des sessions **à l'intérieur du dataset** : retrouver,
parmi les sessions du jour 2, celle qui appartient à chaque identifiant
pseudonymisé du jour 1. On ne cherche jamais à relier un participant à une
identité extérieure au dataset. L'angle du projet est de montrer que *les
données neurophysiologiques se comportent comme une donnée biométrique*, donc
qu'une pseudonymisation par identifiant ne suffit pas à les protéger.

Conditions d'accès au dataset (Hugging Face) : pas de tentative de
ré-identification des participants, pas de redistribution d'enregistrements ni
de dérivés permettant d'identifier un participant, suppression de la copie à la
fin du projet. Le dépôt GitHub est **public**. Conséquences pour le dépôt :

- on ne commite ni données, ni features, ni embeddings, ni matrices de scores
  par participant (`data/`, `cache/`, `out/`, `*.npz`, `*.parquet` ignorés) ;
- `results/inventory.csv` ne contient que ce que la liste publique des fichiers
  montre déjà (identifiants, jours, tâches, durées). Cohorte, âge, genre et nom
  de protocole sont joints à l'exécution depuis `participants_train.csv` et les
  `metadata.json` (`reid.inventory.load_inventory`) ;
- on ne commite que des métriques agrégées, des splits (identifiants
  pseudonymisés, déjà publics dans la liste des fichiers) et des figures
  agrégées ;
- le contrôle « candidats doublons » (identifiants différents, même personne)
  ressemble à de la mise en relation d'enregistrements : **non implémenté tant
  que HABS ne l'a pas validé**, et sa sortie ne sera jamais commitée.

## Environnement

Python 3.12 (`uv venv --python 3.12 .venv`), `requirements.txt`, plus pour le
développement : `scikit-learn==1.9.1 matplotlib==3.10.8 pytest==9.2.0
ruff==0.17.5 huggingface_hub==2.4.0`. Tests : `.venv/bin/python -m pytest tests/reid`.

---

## 2026-11-14 — Étape 0 : inventaire

**Objectif** : connaître la taille réelle du problème avant d'écrire du code
d'évaluation.

**Ce qui a été fait** : branche `reid/baselines`, arborescence du projet
(modules vides avec docstring), `reid/config.py`, `reid/inventory.py`,
`scripts/reid_inventory.py`, `tests/reid/test_inventory.py` (5 tests).
L'inventaire se construit depuis un dossier local (`--data`, lit les
`metadata.json`) ou depuis la liste publique des fichiers Hugging Face (`--hf`,
noms et tailles uniquement, sans authentification). `results/inventory.csv`
vient de la liste HF : cohorte et protocole inconnus tant que les tables ne sont
pas téléchargées ; durées estimées à partir de la taille des BDF
(`duration_source = bdf_size`, ajustement linéaire calé sur les deux BDF de P001).

**Choix et pourquoi** :
- âge et genre retirés de l'inventaire (dépôt public, voir le cadre éthique) ;
- durées estimées depuis la taille des fichiers : suffisant pour planifier
  (repérer les sessions de repos tronquées), remplacé par les durées exactes
  des `metadata.json` après téléchargement ;
- jeu de données par défaut : `CURATED` + notre propre prétraitement.
  `PREPROCESSED` (documenté dans `docs/Dataset_Paper.pdf`, section
  Preprocessing) retire T9/T10, filtre 0,5–48 Hz, référence moyenne, puis ICA
  Picard + ICLabel avec rejet des composantes oculaires (p ≥ 0,75) et
  musculaires / bruit de ligne / bruit de canal (p ≥ 0,90). Le nombre de
  composantes retirées varie selon la session, ce qui peut créer ou effacer de
  l'information propre à la session. À garder comme ablation, pas comme défaut.

**Résultats** (liste HF, `CURATED`) : 355 participants, 596 sessions.

| jours disponibles | 1 | 2 | 1+2 | 1+3 | 2+3 | 1+2+3 |
|---|---|---|---|---|---|---|
| personnes | 75 | 61 | 191 | 4 | 2 | 22 |

Soit 136 personnes à un seul jour et 219 à au moins deux jours.

| scénario | repos sur les deux jours | artefact | vidéo | au moins une tâche |
|---|---|---|---|---|
| d1 → d2 | 208 | **0** | 213 | 213 |
| d1 → d3 | 22 | 0 | 26 | 26 |
| d2 → d3 | 23 | 0 | 24 | 24 |

Hasard top-1 pour d1 → d2 avec une galerie complète : 1/213 ≈ 0,5 %.

**Écarts avec le cahier des charges** :
- 355 participants / 596 sessions dans la release, contre 383 / 690 dans le
  papier : la différence correspond aux 29 participants held-out (sans
  enregistrements). `PREPROCESSED` en compte 354.
- **Le bloc artefact n'existe qu'une fois par personne** (222 sessions : 190 au
  jour 1, 32 au jour 2, jamais aux deux). Pas de comparaison artefact → artefact
  entre jours : ce bloc ne peut servir qu'en transfert entre tâches (par ex.
  repos EO du bloc artefact contre repos du jour suivant) ou comme données
  d'entraînement.
- 6 sessions sans repos (4 au jour 1 avec artefact + vidéo, 2 avec vidéo seule).
- 61 personnes n'ont que le jour 2. Hypothèse : leur jour 1 n'a pas été publié.
  À ne pas creuser par appariement (voir le cadre éthique).
- Durées de repos : 91–99 s (médiane 95,6 s), donc les 30 s EO et 60 s EC sont
  normalement complètes ; 8 sessions sous 92 s à vérifier.
- Sur P001 (vérifié sur les BDF) : 84 colonnes, 64 EEG en µV, ECG en « mV »,
  Resp en AU, GSR en µS après conversion, 17 colonnes d'oculométrie stockées à
  166 Hz (fréquence effective 166,36 Hz, ~83 % de NaN sur la grille 1 kHz, ce qui
  est normal). Distance inter-oculaire : médiane 63,00 mm, écart interquartile
  0,07 mm (repos) et 0,04 mm (artefact). Confirmé.
- **Unité de l'ECG suspecte** : écart-type de 156 « mV » au repos (un ECG fait
  environ 1 mV). Probablement des µV ou des unités brutes : à vérifier avant les
  features d'amplitude (le battement moyen sera normalisé de toute façon).
- Repos de P001 : 10002 (30 s, yeux ouverts *sans cligner*) au début du segment,
  puis 10004 (60 s, yeux fermés) à +35,1 s. Artefact : 10000 (60 s) ×2 et
  10006–10018 (5 s). Codes conformes au cahier des charges sur P001 ; stabilité
  entre cohortes à vérifier sur les `metadata.json`.

**Interprétation** : le scénario principal d1 → d2 porte sur environ
210 personnes avec repos et vidéo. Les scénarios avec le jour 3 sont petits
(environ 25 personnes) : IC larges attendus.

**Limites / à vérifier** : cohortes inconnues tant que `participants_train.csv`
n'est pas téléchargé (accès HF non encore accordé sur cette machine) ; durées
estimées.

**Prochaine étape** : télécharger tables + `metadata.json` + repos + artefact,
régénérer l'inventaire avec `--data ./data`, puis étape 1 (pipeline d'évaluation).

---

## 2026-11-15 — Décisions : CURATED, pas d'oculométrie, inventaire public minimal

**Objectif** : figer les choix de données avant le pipeline d'évaluation.

**Ce qui a été fait** : `reid/config.py` (modalités, fenêtres, bandes),
suppression de `features/gaze.py` et `features/anatomy.py`, inventaire public
réduit aux colonnes issues de la liste des fichiers.

**Choix et pourquoi** :
- `CURATED` + notre propre prétraitement (décision de l'équipe).
- **Oculométrie entièrement retirée** (regard, pupille, centres des globes
  oculaires, axes optiques) : le flux n'est pas synchronisé de façon fiable avec
  les autres. Conséquence : le contrôle « anatomie » (distance inter-oculaire)
  disparaît aussi. Les mouvements oculaires restent visibles dans l'EEG
  frontal (clignements, saccades) : on ne les retire pas, on le garde en tête
  en interprétant les résultats, en particulier sur le bloc artefact.
- Le nom du protocole est retiré de l'inventaire public : `PerfumFemme2025`
  (P001) laisse deviner le genre.

**Limites / à vérifier** : sans oculométrie, les modalités sont EEG, ECG, EDA et
respiration.

**Prochaine étape** : pipeline d'évaluation commun (étape 1).

---

## 2026-11-15 — Étape 1 : pipeline d'évaluation commun

**Objectif** : un seul chemin d'évaluation pour tous les modèles de l'équipe,
avec les mêmes plis, des IC et des contrôles.

**Ce qui a été fait** : `reid/eval/` (`protocol.py`, `interface.py`,
`matching.py`, `metrics.py`, `controls.py`, `runner.py`, `report.py`,
`README.md`), `scripts/reid_run.py`, `scripts/reid_report.py`,
`tests/reid/test_eval.py` + `tests/reid/synthetic.py` (8 tests).

**Choix et pourquoi** :
- Unité commune : la **fenêtre de 10 s** (la durée d'un essai vidéo). Les
  features en cache ont une ligne par fenêtre, ce qui permet les moitiés de
  session (plafond même-jour) et les ablations de durée sans recalcul.
- `fit` ne voit aucune session des personnes de test, même pas leur galerie :
  c'est plus strict que demandé et plus simple à vérifier. L'enrôlement se fait
  dans `score`.
- Les personnes à un seul jour sont réparties dans les plis : elles servent de
  distracteurs pour leur pli et d'entraînement pour les quatre autres (au lieu
  d'être « toujours » en entraînement), pour que les distracteurs soient
  inconnus du modèle.
- Ex-æquo départagés au hasard en espérance, pour que les scores grossiers
  (démographie) ne soient ni avantagés ni pénalisés.
- `folds.json` n'est **pas encore généré** : il sera créé une seule fois, quand
  `participants_train.csv` permettra de stratifier par cohorte.

**Résultats** (synthétiques uniquement) : embeddings parfaits → top-1 = 1 ;
embeddings aléatoires et labels mélangés → hasard dans l'IC ; Hongrois > plus
proche voisin sur un cas construit ; aucune personne de test dans `fit`.

**Interprétation** : le contrôle démographique atteint 0,81 de top-1 sur les
données synthétiques, parce que l'âge exact et le genre sont fixes par personne.
Sur les vraies données, il sera aussi probablement bien au-dessus du hasard :
c'est la barre à battre, pas le hasard seul.

**Limites / à vérifier** : galeries d'environ 43 personnes avec 5 plis ;
courbe de taille de galerie limitée à la taille d'un pli.

**Prochaine étape** : extraction des features (fin de l'étape 0), puis
baselines.

---

## 2026-11-15 — Étape 0 (suite) : extraction des features en cache

**Objectif** : calculer une fois pour toutes, par session, les features dont
toutes les baselines ont besoin.

**Ce qui a été fait** : `reid/segments.py` (sous-segments depuis
`metadata.json`, lecture paresseuse des canaux utiles avec MNE),
`reid/preprocess.py`, `reid/features/` (`eeg_psd`, `eeg_cov`, `ecg`, `eda`,
`resp`, orchestrateur dans `__init__.py`), `scripts/reid_extract.py`,
`process.py` + `reid_job.yaml`, `requirements.txt` (scikit-learn, pyriemann,
neurokit2, pyyaml, matplotlib), `tests/reid/test_extract.py`.
`habs run --data ./sample --out ./out` écrit `features/rest/P001_day-1.npz` et
`features/artefact/P001_day-1.npz`.

**Choix et pourquoi** :
- **`process.py` remplacé** : l'exemple « puissance alpha » du template laisse
  place à l'extraction pilotée par `reid_job.yaml` (tâches, modalités, shard
  `i/n` pour rester sous une heure par job).
- Lecture fenêtre par fenêtre avec `mne.io.read_raw_bdf(preload=False)` : on ne
  charge jamais le DataFrame complet (indispensable pour les vidéos).
- Fenêtres de 10 s dans chaque sous-segment : repos EO (3), EC (6), repos du bloc
  artefact (12), une fenêtre par action de 5 s (7), un essai vidéo = une fenêtre.
- EEG : **T9 et T10 exclus** (62 électrodes), comme dans `PREPROCESSED`. Chez
  P001, T10 a une amplitude crête-à-crête médiane de 917 µV contre 65 µV pour
  les autres électrodes. Il faisait monter le seuil de rejet à 1 316 µV ; sans
  T9/T10, le seuil est de 276 µV et 3,7 % des époques sont rejetées.
- Filtre Butterworth d'ordre 4 à phase nulle, avec 2 s de signal de part et
  d'autre de chaque fenêtre pour absorber les effets de bord.
- PSD : périodogramme de Hann par époque de 2 s, recouvrement 50 %, moyenne des
  époques propres = Welch. Résolution 0,5 Hz, 1–45 Hz, en µV²/Hz.
- Covariances OAS par époque et par bande (1–45, thêta, alpha, bêta), puis
  moyenne riemannienne des époques propres de la fenêtre.
- ECG, EDA, respiration : `neurokit2`, NaN en cas d'échec. EDA et respiration
  utilisent ±10 s de contexte (une fenêtre de 10 s contient environ trois
  respirations). L'unité de l'ECG étant douteuse, on n'utilise que des features
  sans unité : FC, SDNN, RMSSD, battement moyen normalisé.

**Résultats** (P001 jour 1) : repos 9 fenêtres en 1,8 s ; artefact 19 fenêtres
en 4,2 s ; FC ≈ 80 bpm, respiration ≈ 18/min, EDA ≈ 2,4 µS. La fenêtre « tête
haut-bas » n'a aucune époque propre (NaN), ce qui est attendu.

**Limites / à vérifier** : pas de détection de canaux défectueux au-delà de
T9/T10 ; features lentes bruitées sur 10 s ; taille du cache estimée à environ
2 Go pour toutes les tâches.

**Prochaine étape** : baselines 1 à 3.

---

## 2026-11-15 — Étape 2 : baseline 1, corrélation des spectres (code seul)

**Objectif** : un plancher sans apprentissage, à battre par tout le reste.

**Ce qui a été fait** : `reid/baselines/common.py` (assemblage des features,
z-score, Pearson), `reid/baselines/b1_spectral_corr.py` (`CorrModel`), 9
variantes EEG (`abs`/`rel`/`bands` × EC/EO/EC+EO) et 4 variantes « une modalité »
pour l'étape 4 (`ecg`, `ecg_beat`, `eda`, `resp`), `tests/reid/test_baselines.py`
avec un générateur synthétique au format du cache.

**Choix et pourquoi** :
- Spectre de session = log de la PSD moyenne des époques propres (pondérée par
  le nombre d'époques par fenêtre), c'est-à-dire le Welch de toute la condition.
- z-score avec les sessions d'entraînement du pli uniquement ; valeurs
  manquantes → 0 après z-score (neutre pour la corrélation).
- Le même `CorrModel` sert pour les autres modalités (moyenne des fenêtres).

**Résultats** : pas encore de données réelles. Sur synthétique : top-1 > 0,8
avec identité, hasard sans identité ; avec un fort gain d'électrode par jour,
`rel` bat `abs`, comme attendu.

**Prochaine étape** : baseline 2 (covariances riemanniennes).

---

## 2026-11-15 — Étape 3 : baseline 2, covariances riemanniennes (code seul)

**Objectif** : tester si la structure spatiale de l'EEG (covariance entre
électrodes, liée à la géométrie de la tête) identifie mieux que le spectre.

**Ce qui a été fait** : `reid/baselines/b2_riemann.py` (`RiemannModel`), 13
variantes (`airm`/`tangent` × large bande/filter bank × EC/EO/EC+EO, plus le
contrôle `_recentered`), tests synthétiques.

**Choix et pourquoi** :
- Covariance de session = moyenne riemannienne des covariances de fenêtres
  (elles-mêmes moyennes riemanniennes des époques OAS), par condition et par
  bande.
- `tangent` : point de référence et z-score appris sur les sessions
  d'entraînement du pli uniquement.
- `airm` : distances au carré sommées sur conditions × bandes.
- Contrôle `_recentered` : chaque session est blanchie par sa propre
  covariance moyenne. Il reste seulement le contraste EC/EO, et l'identité doit
  disparaître.
- Les 62 électrodes : la distance d'un pli complet (45 × 45) prend environ
  0,2 s, donc pas besoin du sous-ensemble de 32 électrodes.

**Résultats** (synthétiques) : les deux métriques retrouvent l'identité
(top-1 = 1,0 sans gain par jour). Le recentrage l'efface. **Observation à
vérifier sur les vraies données** : avec un gain d'électrode qui change d'un jour
à l'autre (écart-type 0,3 en log10), `airm` tombe à 0,40 alors que `tangent`
reste à 0,96. `airm` pèse toutes les directions de la même façon, alors que le
z-score de l'espace tangent apprend à ignorer celles qui bougent avec
l'impédance.

**Prochaine étape** : baseline 3 (PCA + LDA, puis PLDA).
