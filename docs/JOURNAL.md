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

---

## 2026-11-15 — Étape 5 : baseline 3, PCA + LDA puis PLDA (code seul)

**Objectif** : apprendre à ignorer ce qui change d'un jour à l'autre.

**Ce qui a été fait** : `reid/baselines/b3_lda_plda.py` (`LdaModel`,
`TwoCovPLDA`), 4 variantes (`b3_lda_eeg`, `b3_lda_all`, `b3_plda_eeg`,
`b3_plda_all`), tests synthétiques dont une vérification de la PLDA contre la
log-vraisemblance calculée par force brute (`scipy.stats.multivariate_normal`).

**Choix et pourquoi** :
- Entrée par fenêtre de 10 s : spectre EEG relatif (62 × 89), ECG (3 + battement
  moyen 128), EDA (2), respiration (3), soit 5 654 features. Imputation par la
  médiane → standardisation → PCA → LDA (`eigen`, `shrinkage="auto"`), une
  classe par personne d'entraînement **ayant au moins deux jours**.
- Nombre de composantes PCA (32/64/128/256) choisi par validation interne :
  20 % des personnes d'entraînement ayant les jours 1 et 2, top-1 jour 1 →
  jour 2.
- Embedding de session = moyenne des projections de ses fenêtres → cosinus.
- PLDA à deux covariances ajustée sur les embeddings de session des personnes
  d'entraînement (moments : W intra-personne, B = covariance des moyennes − W/n),
  normalisation de longueur, score = log-rapport de vraisemblance exact.
  Log-déterminants par Cholesky (numpy/Accelerate émettait de faux warnings avec
  `slogdet`).
- Un ajustement PCA 256 + LDA à la taille réelle prend environ 1,6 s.

**Résultats** (synthétiques) : la PLDA bat nettement le cosinus quand le bruit
de session est concentré là où les personnes ne diffèrent pas. Avec un fort gain
par jour, la LDA fait au moins aussi bien que la baseline 1 `abs`.

**Limites / à vérifier** : la PLDA est ajustée sur les embeddings des sessions
qui ont servi à la LDA, ce qui sous-estime W (biais classique, aussi présent en
vérification du locuteur). Si c'est gênant, il faudra l'ajuster sur la PCA ou
sur des données mises de côté.

**Prochaine étape** : baseline 5 (réseau siamois), puis tout lancer sur les
vraies données.

---

## 2026-11-15 — Étape 7 : baseline 5, réseau siamois (code seul)

**Objectif** : tester si une projection non linéaire fait mieux que la LDA.

**Ce qui a été fait** : `reid/baselines/b5_siamese.py` (`SiameseModel`,
`supcon_loss`), 3 seeds enregistrées (`b5_siamese_s0..s2`), tests (la perte
n'utilise que les positifs entre jours différents ; identité retrouvée sur
synthétique). Correction de `make_folds` quand un groupe a moins de personnes
que de plis.

**Choix et pourquoi** :
- Entrée = features standardisées de l'étape 5 (avant PCA) ; MLP
  256 → 128 → 64, dropout 0,2, sortie normalisée L2.
- SupCon : positifs = même personne, **jour différent** ; les paires même
  personne + même jour sont retirées du dénominateur.
- Lots : 32 personnes × 2 jours × 4 fenêtres. Augmentations : bruit gaussien,
  masquage de blocs de modalité (p = 0,2), décalage par électrode sur le bloc
  EEG (simule l'impédance).
- Arrêt précoce (patience 30 itérations) sur les mêmes 20 % de personnes de
  validation interne que la LDA. Résultat à rapporter = moyenne des 3 seeds.
- `torch` n'est pas ajouté à `requirements.txt` : il est déjà dans l'image de la
  plateforme.

**Essai de bout en bout sur le vrai format de cache** : le fichier de P001
dupliqué en 24 « personnes » × 2 jours, seule différence = un gain par
électrode. Les 31 modèles tournent sans erreur. `abs` et covariances retrouvent
l'identité, `rel`/`bands`/LDA sur `rel` sont au hasard, comme attendu (le
spectre relatif annule les gains). **Nuance** : le contrôle « recentrage par
session » reste à 1,00 dans ce cas. Blanchir une session par sa propre moyenne
retire la moyenne mais laisse une rotation qui dépend des gains d'électrode :
le recentrage n'efface pas forcément toute l'identité. À regarder sur les vraies
données.

**Prochaine étape** : code des ablations (étape 6), puis tout lancer dès que
les données sont là.

---

## 2026-11-15 — Étape 6 : ablations (code seul)

**Objectif** : produire les figures principales du papier (modalité, tâche,
durée, émotion) avec la meilleure méthode, quand les données seront là.

**Ce qui a été fait** : `reid/baselines/b4_ablations.py`,
`scripts/reid_ablations.py`, figures d'ablation dans `reid/eval/report.py`
(heatmap modalités, barres tâches/arousal, courbe durée), paramètre `part`
(N premières secondes de la galerie et de la probe) et conditions qui dépendent
de la tâche dans le pipeline. Test de bout en bout sur synthétique.

**Choix et pourquoi** :
- Durée = N premières secondes **de chaque condition**, en galerie comme en
  probe (repos : 10/20/30/60 s ; vidéo : 10 à 360 s). L'entraînement garde
  toutes ses fenêtres.
- Transfert entre tâches : l'entraînement voit les deux tâches. Sans cela, la
  LDA n'aurait aucune personne avec le bloc artefact sur deux jours.
- Repos EO du bloc artefact (jour 1) → repos EO du jour 2 : seul usage possible
  du bloc artefact entre deux jours. **Les actions (clignements, saccades,
  mâchoire, déglutition) ne peuvent pas être comparées d'un jour à l'autre** :
  le bloc n'existe qu'un jour par personne.
- Arousal : découpage à la médiane de chaque session (les échelles diffèrent
  entre cohortes : entiers en cohorte 1, continu ailleurs), essais repérés par
  `start_marker`.

**Limites / à vérifier** : la correspondance `start_marker` ↔ `w_marker` et le
format de la colonne `day` de `labels_train.csv` sont à vérifier sur les vraies
tables.

**Prochaine étape** : données → inventaire exact → `folds.json` → extraction →
lancer les baselines.

## 2026-11-16 — Données : accès, vérifications, plis partagés

**Objectif** : récupérer CURATED (repos + artefact), vérifier les hypothèses
faites sans données, figer les plis de toute l'équipe.

**Ce qui a été fait** :
- Accès HF accordé. Commande correcte (un `--include` **par motif** : avec
  `--include a b c`, `hf` lit b et c comme des noms de fichiers) :
  `hf download HABSteam/multimodal-eeg-emotion --repo-type dataset --local-dir ./data --include "*.csv" --include "CURATED/*/*_metadata.json" --include "CURATED/*/*.record_map" --include "CURATED/*/*resting-state*" --include "CURATED/*/*artefact*"`.
- Connexion très lente ce jour-là (~120 ko/s, mesuré aussi hors HF) :
  téléchargement relancé sous watchdog, repos en priorité.
- Marqueurs vérifiés sur les 596 métadonnées (5 cohortes) : 590 sessions de
  repos ont toutes 3 fenêtres EO + 6 EC, 222 sessions artefact ont toutes
  12 fenêtres EO + les 7 actions. Aucune exception.
- `labels_train.csv` : `day` est un flottant (1.0/2.0/NaN), `start_marker` un
  entier qui correspond aux marqueurs vidéo. **Bug corrigé** dans
  `arousal_split` (1.0 devenait le jour 10) + test de non-régression.
- `participants_train.csv` : 383 lignes ; P057 (une session) n'y figure pas
  (cohorte inconnue, gardé, non stratifié).
- `results/splits/folds.json` : 5 plis sur les 355 personnes de l'inventaire,
  stratifiés par cohorte ; ~44 personnes multi-jours de test + ~27
  distracteurs par pli. Ne contient que des identifiants déjà publics.

**Choix et pourquoi** : plis tirés sur l'inventaire complet (et non sur les
fichiers déjà téléchargés), pour qu'ils ne changent pas quand le reste arrive.

## 2026-11-16 — Correctifs trouvés sur les vraies données

**Ce qui a été fait** (premier passage sur ~240 personnes déjà téléchargées) :
- **Sessions à 998 Hz** (P029, P066, P108, …) : la grille du périodogramme de
  2 s n'était plus exactement à 0,5 Hz, 88 bins au lieu de 89 → les modèles
  plantaient (vecteurs de tailles différentes). Les spectres sont maintenant
  interpolés sur une grille fixe de 89 bins (identité exacte à 1000 Hz, donc
  seules les sessions à 998 Hz ont été réextraites). `VERSION = 2` dans les npz.
  Test : même grille et même pic à 998, 1000 et 1024 Hz.
- **Contrôle démographique** : 3 sessions sans genre (`pd.NA`) faisaient planter
  la comparaison. Genre manquant = pas de pénalité. Test ajouté.
- **B2 lent** (253 s pour `b2_tangent_fb`) : 90 % du temps dans les moyennes
  riemanniennes par session, recalculées à chaque pli. Mémoïsées par contenu
  (hash des covariances, pas les identifiants, que les tests synthétiques et
  les ablations réutilisent) : 76 s, résultats identiques.

**Constats** :
- Le contrôle démographique est **très fort** : âge en années entières, un
  couple (âge, genre) regroupe en médiane 4 personnes du dataset entier, donc il
  est souvent unique dans une galerie de ~30-45 personnes. Premier passage :
  top-1 ≈ 0,75, EER ≈ 0,01. C'est la barre à battre, pas le hasard.
- Features physiologiques plausibles (FC 55-109 bpm, EDA tonique 1-9 µS,
  respiration ~16 /min, 6 % d'epochs EEG rejetées en médiane).
- **Contrôle « cohorte prévisible »** : il empilait les embeddings de tous les
  plis, alors que chaque pli a son propre modèle (espaces différents, et même
  tailles différentes pour la LDA, d'où un plantage). Calculé maintenant pli
  par pli (galerie du pli), moyenne pondérée. Test de non-régression.
- **PLDA** : sur le pli 0, top-1 PLDA 0,29 contre 0,93 au cosinus. Diagnostic :
  la PLDA s'effondre quand la dimension approche le nombre de personnes
  d'entraînement (~170) — 0,81 en 32 dimensions, 0,73 en 64, 0,29 en 172 —
  alors que le choix de dimension se faisait au cosinus (qui préfère 256). La
  validation interne utilise maintenant le score du modèle lui-même (PLDA
  ajustée sur les personnes internes) ; grille 16-256.
- Téléchargement terminé : 590 sessions repos + 222 artefact (21 Go), toutes
  extraites sans échec.

## 2026-11-16 — Premiers résultats : baselines B1-B5 et ablations (repos)

**Objectif** : faire tourner les 34 modèles enregistrés sur les 6 scénarios
(repos, plis partagés) et les ablations de la meilleure méthode.

**Ce qui a été fait** : 204 runs + 18 ablations (`results/runs/`, agrégats
seulement : ni identifiant, ni âge, ni genre), `results/leaderboard.csv`,
figures dans `results/figures/`. Deux correctifs en route :
- **Appariement hongrois et égalités** : `linear_sum_assignment` garde le
  premier optimum ; galerie et sondes étant triées par le même identifiant, la
  diagonale (la bonne réponse) gagnait toutes les égalités. Seul le contrôle
  démographique (scores entiers) était touché : 1,00 → 0,78. Égalités tirées au
  hasard (moyenne de 20 permutations), test ajouté ; les modèles à scores
  continus ne changent pas (vérifié sur `b1_abs_ec+eo`).
- Figures : courbe de taille de galerie = meilleur modèle par famille + la
  démographie (et non trois ablations quasi identiques) ; légende des barres
  sous le graphique.

**Résultats (d1 → d2, 208 sondes, galerie ~42, hasard 0,024 ; IC 95 % bootstrap
sur les personnes)** :

| modèle | top-1 | IC 95 % | top-5 | hongrois | EER |
|---|---|---|---|---|---|
| b3_lda_all (EEG+ECG+EDA+resp) | 0,83 | 0,77-0,88 | 0,95 | 0,93 | 0,08 |
| b3_lda_eeg | 0,78 | 0,73-0,84 | 0,91 | 0,85 | 0,11 |
| b3_plda_all | 0,74 | 0,67-0,80 | 0,91 | 0,81 | 0,12 |
| **contrôle démographique** (âge, genre) | 0,71 | 0,67-0,75 | 1,00 | 0,78 | 0,01 |
| b1_abs_ec+eo (corrélation des spectres) | 0,66 | 0,60-0,72 | 0,86 | 0,70 | 0,14 |
| b1_ecg_beat (forme du battement seule) | 0,60 | 0,53-0,66 | 0,80 | 0,57 | 0,19 |
| b5_siamese (meilleure graine) | 0,60 | 0,53-0,66 | 0,84 | 0,67 | 0,21 |
| b2_tangent_fb_ec (riemannien) | 0,52 | 0,45-0,59 | 0,80 | 0,59 | 0,17 |
| b2_airm_broad_ec | 0,20 | 0,15-0,26 | 0,45 | 0,41 | 0,35 |
| b1_ecg / eda / resp (FC, HRV, EDA, resp.) | 0,07-0,09 | | | | |

Autres scénarios pour b3_lda_all : avec distracteurs 0,81 (démographie 0,56) ;
dans la cohorte 0,91 (0,88) ; plafond même jour 1,00 (0,70) ; d1 → d3 1,00 et
d2 → d3 0,87, mais 22-23 sondes seulement (hasard 0,22), IC très larges.

Ablations (b3_lda, toutes modalités) :
- modalités : EEG seul 0,78, forme du battement ECG seule 0,69, FC/HRV,
  EDA, respiration seules ≈ 0,10 ; sans EEG 0,72 (porté par le battement) ;
  retirer FC/HRV, EDA ou respiration ne change rien.
- tâche : EC 0,80 > EO 0,74 ; EC+EO 0,83 ; repos EO du bloc artefact (j1) →
  repos EO (j2) 0,68.
- durée par condition : 10 s 0,71 ; 20 s 0,75 ; 30 s 0,81 ; 60 s 0,83.
- vidéo / arousal : non lancées (données vidéo, 152 Go, non téléchargées).

**Lecture** :
- La LDA (apprendre ce qui reste stable d'un jour à l'autre) est nettement la
  meilleure ; la PLDA ne l'améliore pas avec ~170 personnes d'entraînement ; le
  siamois sous-performe la LDA (peu de données, pas encore réglé).
- **La démographie seule est la vraie barre** : sur 42 personnes, elle est à
  0,71 ; seules les LDA la dépassent, et l'écart grandit avec la galerie
  (`gallery_size.png`) et avec des distracteurs.
- **La forme du battement cardiaque est un identifiant presque aussi fort que
  l'EEG** (biométrie ECG connue) ; les marqueurs autonomes (FC, HRV, EDA,
  respiration) varient trop d'un jour à l'autre pour identifier.
- Les embeddings LDA ne prédisent pas la cohorte mieux que la classe majoritaire
  (0,44 contre 0,44) : pas de confusion cohorte ↔ identité visible.
- AIRM brut s'effondre (0,20) alors que l'espace tangent tient (0,41-0,52) : la
  standardisation apprise sur l'entraînement absorbe les gains du jour, comme
  sur synthétique. Le recentrage par session n'efface pas l'identité (0,50).

**Limites** : d → d3 trop peu nombreux pour conclure ; un seul découpage en
5 plis ; hyperparamètres du siamois non réglés ; ECG en unités du header.

---

## 2026-11-19 — Modèle final, étapes 1-2 : cache brut et un encodeur par signal

**Objectif** : préparer le modèle final (EEG, forme du battement cardiaque,
respiration ; EDA, FC/HRV et oculométrie écartées) à partir des signaux bruts,
et vérifier que chaque encodeur entraîné seul bat la LDA sur le même signal.

**Ce qui a été fait** :
- `reid/final/raw.py` + `scripts/reid_extract.py --raw` : cache
  `cache/raw/<tâche>/<PID>_day-<d>.npz` (1,2 Go, jamais commité). Par
  sous-segment (repos yeux fermés / ouverts, blocs yeux ouverts de la tâche
  artefact) : EEG référence moyenne, 1-45 Hz, 128 Hz, float16 ; battements ECG
  individuels (-250/+450 ms, 128 points) et leurs instants ; respiration
  nettoyée à 25 Hz ; seuil crête-à-crête de la séance (médiane + 5 MAD des
  époques de 2 s). 590 repos + 222 artefact, aucun échec. Rééchantillonnage
  polyphase exact pour 998 / 1000 / 1024 Hz.
- `reid/final/crops.py` (fenêtres), `encoders.py`, `model.py`
  (`ContrastiveModel`, interface `ReIDModel` commune) ; modèles `f_<signal>_s<k>`
  dans `scripts/reid_run.py`, qui lisent le cache brut.
- 9 tests (`tests/reid/test_final.py`) : extraction sur P001, fenêtres qui
  évitent un artefact, battements aberrants retirés, identité retrouvée sur
  données synthétiques brutes (EEG et battement).

**Choix et pourquoi** :
- Repos seulement (EC + EO, 90 s par séance) : même épreuve que les baselines,
  comparaison directe avec la LDA.
- Fenêtres : EEG 4 s (tirage aléatoire sur une grille de 0,25 s, gardée si
  toutes ses époques de 2 s passent le seuil) ; ECG 16 battements consécutifs
  normalisés un à un (forme seule, les unités varient), battements corrélés
  < 0,5 au battement médian retirés ; respiration 20 s normalisée. Embedding
  de séance = moyenne des fenêtres sur une grille fixe (2 s, 8 battements, 5 s).
- Encodeurs petits (~10⁴-10⁵ paramètres, ~170 personnes à deux jours par pli) :
  EEG = spectre de Welch de chaque électrode (relatif + puissance totale) comme
  jeton, plongement d'électrode appris, transformer 2 couches entre électrodes,
  masquage aléatoire de 10 % des électrodes ; ECG = CNN 1D par battement,
  moyenne sur le groupe ; respiration = CNN 1D, moyenne + max.
- Perte SupCon, positifs = même personne **autre jour** ; les personnes venues
  un seul jour servent de négatifs supplémentaires. Lots de 32 personnes × 2
  jours × 4 fenêtres + 16 personnes d'un jour. AdamW, montée puis décroissance
  cosinus, arrêt précoce sur les mêmes personnes de validation interne que la
  LDA. Augmentations : gain par électrode (~10 %), décalage du pic R ± 2
  points, bruit.
- Calcul local sur le GPU du Mac (MPS) : ~30 s par pli et par signal.

**Correctif** : 12 séances ont des pics > 65 mV qui débordaient du float16
(valeurs infinies → embeddings NaN). L'EEG est écrêté à ± 60 mV à l'extraction
(VERSION = 2) ; ces passages échouent de toute façon au test d'amplitude.

**Résultats (d1 → d2, repos, 208 sondes, hasard 0,024, graine 0)** :

| signal | encodeur | top-1 [IC 95 %] | top-5 | EER | LDA même signal |
|---|---|---|---|---|---|
| EEG | `f_eeg_s0` | 0,68 [0,62-0,74] | 0,87 | 0,15 | **0,78** |
| battement ECG | `f_ecg_beat_s0` | 0,63 [0,56-0,69] | 0,86 | 0,15 | **0,69** |
| respiration | `f_resp_s0` | **0,38** [0,32-0,45] | 0,72 | 0,24 | 0,09 |

**Lecture** :
- La respiration passe de 0,09 à 0,38 : la *forme* de l'onde respiratoire
  identifie, alors que ses statistiques (fréquence, variabilité, I/E) non. À
  vérifier : identité réelle ou empreinte du capteur (ceinture, serrage) ?
- EEG et battement restent sous la LDA. L'arrêt précoce tombe très tôt (pas
  100-600 sur ~1000) : le réseau surapprend vite, la validation interne (33
  personnes) est bruitée et optimiste (0,70-0,85 contre 0,68 en test).
- Aucun embedding ne prédit la cohorte mieux que la classe majoritaire.

**Limites** : une seule graine ; aucun réglage ; les réglages à venir se feront
sur la validation interne seulement (pas sur les plis de test).

**Prochaine étape** : régler EEG et battement sur la validation interne (durée
de fenêtre, puissance totale ou non, régularisation), puis fusion.

---

## 2026-11-19 — Modèle final, étape 2 bis : réglage sur validation interne

**Objectif** : rapprocher les encodeurs EEG et battement de la LDA sans
regarder les plis de test.

**Méthode** : chaque configuration est entraînée sur les 5 plis et jugée par
la moyenne du top-1 de **validation interne** (mêmes 20 % de personnes que la
LDA) ; les plis de test ne sont lancés qu'une fois, sur la configuration
retenue. Script de balayage hors dépôt (bloc-notes de session).

**Validation interne (moyenne 5 plis, ~33 personnes par pli)** :

| configuration | EEG | battement |
|---|---|---|
| LDA, même découpage (meilleure dimension, donc flattée) | 0,817 | 0,756 |
| encodeur de départ (fenêtres isolées, 32 personnes / lot) | 0,780 | 0,732 |
| sans puissance totale / fenêtre 8 s / plus petit / masquage 30 % / lr 3e-4 | 0,756-0,786 | — |
| 8 ou 32 battements par groupe / lr 3e-4 | — | 0,720-0,738 |
| moyenne de 4 fenêtres par (personne, jour) avant la perte | 0,793 | 0,738 |
| **moyenne de 8 fenêtres, 64 personnes / lot** | **0,841** | 0,702 |
| 8 fenêtres non moyennées, 64 personnes / lot | 0,762 | — |
| moyenne de 16 fenêtres, 64 personnes / lot | 0,848 | — |
| moyenne de 8 fenêtres, 96 personnes / lot | 0,835 | — |

Les hyperparamètres classiques ne changent rien (bruit ± 0,03). Ce qui aide
l'EEG : **apprendre au niveau de la séance** (option `pool_crops` : l'unité de
la perte est la moyenne normalisée de 8 fenêtres d'une personne un jour donné,
comme à l'évaluation), et non la taille des lots. Retenu : 8 fenêtres, 64
personnes (16 fenêtres n'est pas différent et coûte deux fois plus). Rien
n'aide le battement : on garde l'encodeur de départ.

**Plis de test, une seule fois (d1 → d2, repos, 208 sondes)** :

| modèle | top-1 [IC 95 %] | top-5 | hongrois | EER |
|---|---|---|---|---|
| `f_eeg_pool_s0` | 0,70 [0,64-0,76] | 0,90 | 0,84 | 0,13 |
| `f_eeg_pool_s1` | 0,71 [0,64-0,76] | 0,89 | 0,80 | 0,11 |
| `f_eeg_pool_s2` | 0,72 [0,66-0,78] | 0,90 | 0,79 | 0,11 |
| `f_eeg_s0` (départ) | 0,68 [0,62-0,74] | 0,87 | 0,72 | 0,15 |
| `b3_lda_eeg` | **0,78** [0,73-0,84] | 0,91 | 0,85 | 0,11 |

**Lecture** : le gain de validation ne se transfère qu'en partie (+0,03 en
top-1, +0,03 en top-5, EER 0,15 → 0,11). L'encodeur EEG reste sous la LDA.
Un désavantage de protocole reste : le réseau n'est entraîné que sur 80 % des
personnes d'entraînement (les 20 % de validation servent à l'arrêt précoce),
alors que la LDA est réajustée sur toutes une fois sa dimension choisie.

**Prochaine étape** : réentraîner sur toutes les personnes d'entraînement
jusqu'au pas choisi (parité avec la LDA), puis fusion des trois encodeurs.

---

## 2026-11-19 — Modèle final, étape 2 ter : réentraînement sur toutes les personnes

**Objectif** : même traitement que la LDA. Jusqu'ici le réseau n'apprenait que
sur 80 % des personnes d'entraînement (les 20 % de validation servaient à
l'arrêt précoce) ; la LDA est réajustée sur toutes une fois sa dimension choisie.

**Ce qui a été fait** : option `refit` de `ContrastiveModel` : après l'arrêt
précoce, un réseau neuf (même graine, même calendrier de taux d'apprentissage)
est entraîné sur toutes les personnes d'entraînement jusqu'au pas retenu, sans
validation. Décidé avant de voir les résultats (parité de protocole, pas un
réglage). Test : sur synthétique, le réseau réentraîné voit toutes les personnes
et retrouve l'identité (couvre aussi `pool_crops`).

**Plis de test (d1 → d2, repos, 208 sondes ; moyenne de 3 graines, plage)** :

| signal | sans réentraînement | **avec réentraînement** | LDA même signal |
|---|---|---|---|
| EEG (moyenne de 8 fenêtres) | 0,708 (0,70-0,72) | **0,726** (0,72-0,74), EER 0,12 | 0,784 |
| battement ECG | 0,630 (graine 0) | **0,659** (0,65-0,67), EER 0,15 | 0,688 |
| respiration | 0,385 (graine 0) | 0,338 (0,32-0,35), EER 0,23 | 0,087 |

**Lecture** : le réentraînement aide un peu l'EEG et le battement (+0,02 à
+0,03), sans les amener au niveau de la LDA ; la respiration ne change pas au-delà
de la variation entre graines (une seule graine sans réentraînement). La
respiration apprise reste l'apport nouveau : 0,34 contre 0,09 pour ses
statistiques, information que la LDA n'exploite pas. Aucun embedding ne prédit
la cohorte mieux que la classe majoritaire (0,33-0,39 contre 0,44).

**Prochaine étape** : fusion des trois signaux, en comparant (a) un modèle
appris de bout en bout et (b) une fusion des scores LDA (EEG + battement) et de
l'encodeur respiration, poids choisis sur validation interne.

---

## 2026-11-22 — Fusion dans main, modèle final étape 3 : fusion des scores

**Fusion dans main** (avec l'accord explicite) : `reid/baselines` puis
`reid/final` fusionnées dans `main`. Conflit sur `process.py` : la version de
l'équipe (classification) est gardée ; le point d'entrée reid est déplacé dans
`reid/platform.py`. Pour les jobs GPU (la clé ne marche que depuis `main`),
`process.py` délègue temporairement à `reid/platform.py`, le code de l'équipe
restant en commentaire à l'identique (un `git revert` du commit le rétablit).
Signalé à l'utilisateur ; à coordonner avec l'équipe.

**Job GPU** (`reid_job.yaml`, `job: final`) : une job ne lit pas la sortie d'une
autre, donc tout se fait dans une seule : extraction des deux caches (features et
crops bruts) en parallèle dans un scratch local, puis un processus par groupe de
modèles (une graine par groupe) ; runs dans `out_dir/runs`, journaux dans
`out_dir/logs`, matériel et durées dans `out_dir/job.json`. Testé de bout en bout
sur `sample/` (pas de sonde jour 2 dans l'échantillon : les modèles sont sautés).

**Fusion des scores** (`reid/final/fusion.py`, `ScoreFusion`) : tout `ReIDModel`
peut être membre, chacun lisant son propre cache. Les scores de chaque membre
sont z-normalisés par sonde (sur la galerie), puis sommés avec des poids. Poids
« réglés » : simplexe au pas 0,1, top-1 de validation interne (départage MRR puis
proximité des poids égaux), à partir des scores de validation que la LDA (à sa
dimension retenue) et l'encodeur (à son pas d'arrêt) calculent déjà sans les
personnes de validation : pas de second entraînement. Les encodeurs d'une fusion
partagent le split interne de la LDA (`split_seed`). Poids « égaux » : rien de
réglé. Les 7 configurations ci-dessous ont été fixées avant les résultats et
lancées une fois chacune.

**Plis de test (d1 → d2, repos, 208 sondes, chance 0,024 ; 3 graines des encodeurs)** :

| fusion | membres | top-1 (graines 0 / 1 / 2) | moyenne | top-5 | EER |
|---|---|---|---|---|---|
| référence | LDA EEG + battement + resp. (features concaténées) | 0,827 [0,77-0,88] | 0,827 | 0,952 | 0,059 |
| LDA3 + resp. | référence + encodeur resp., réglés | 0,851 / 0,880 / 0,851 | 0,861 | 0,960 | 0,053 |
| LDA2 + resp. | LDA EEG, LDA battement, encodeur resp., réglés | 0,889 / 0,909 / 0,909 | 0,902 | 0,981 | 0,039 |
| 3 encodeurs | EEG, battement, resp., réglés | 0,904 / 0,880 / 0,875 | 0,886 | 0,954 | 0,058 |
| 3 encodeurs | idem, poids égaux | 0,894 / 0,918 / 0,861 | 0,891 | 0,982 | 0,044 |
| 5 membres | LDA EEG, LDA battement, 3 encodeurs, réglés | 0,942 / 0,918 / 0,947 | 0,936 | 0,979 | 0,033 |
| **5 membres** | idem, **poids égaux** | **0,966 / 0,957 / 0,952** | **0,958** | 0,986 | **0,022** |

IC à 95 % de la meilleure : [0,92-0,99] selon la graine, disjoint de celui de la
référence [0,77-0,88]. Contrôles : identités permutées 0,019 (chance 0,024) ;
démographie seule 0,713 ; prédiction de la cohorte 0,35-0,47 pour une classe
majoritaire à 0,44 (aucun embedding ne la prédit mieux).

**Lecture** :
- Les trois encodeurs appris, seuls tous sous la LDA de leur signal, dépassent
  la référence une fois fusionnés (0,89) : leurs erreurs sont peu corrélées.
- LDA et encodeurs se complètent : 5 membres > LDA seules (0,90) et > encodeurs
  seuls (0,89). Les deux familles regardent le même signal autrement (spectre
  par fenêtre et projection linéaire contre attention entre électrodes et
  contraste au niveau de la séance).
- Les poids égaux valent ou battent les poids réglés : 33 sondes de validation
  par pli ne suffisent pas à régler 5 poids. **Réserve** : la variante à poids
  égaux a été désignée meilleure en regardant le test. Elle n'enregistrait pas
  de score de validation ; c'est corrigé (`val_top1_equal`) pour que le choix
  entre variantes se fasse dorénavant sur validation. Les 7 variantes étant
  fixées d'avance, la LDA2 + resp. réglée (0,902) et la 5 membres réglée
  (0,936) sont les chiffres sans aucune sélection sur le test.

**Prochaine étape** : job GPU (reproduction de ces fusions sur H100), puis
pré-entraînement « world model » de l'encodeur EEG sur l'EEG des vidéos
(seulement disponible sur le GPU) ; fusion apprise de bout en bout à comparer à
la fusion à poids égaux.

## 2026-11-22 — Récupérer les résultats d'un job GPU depuis GitHub

Le premier job GPU (`20261122T003121-0fcdc8`) a été soumis par `submit.yml`,
dont l'étape `habs watch` abandonne au bout d'1 h, alors que les résultats
restent dans le stockage de l'équipe (`out/<job>/`). Sans lien de résultats,
on ne pouvait pas les lire.

Ajout de `.github/workflows/results.yml` (hors reid/, signalé) : workflow
manuel, sans GPU ni nouveau job. Il se connecte avec la même clé d'équipe
que `submit.yml` (donc seulement depuis main), affiche `habs status`,
télécharge `out/<job>/` par `habs get` plus la sortie console (`habs logs`),
et joint le tout au run en artefact conservé 7 jours. Le repo étant public,
les listes d'échecs (`failures_*.csv`, qui nomment des participants) sont
retirées de l'artefact ; seul leur nombre de lignes est affiché.

Testé en local : client `habs` sur un stockage simulé (`HABS_STORE_DIR`),
étapes du workflow exécutées telles quelles, avec un id de job et sans id
(dernier job). Le fichier d'échecs est bien écarté, le reste est conservé.

Correctif : sur le job `20261122T003121-0fcdc8`, `habs get` n'a rien
téléchargé (le job n'avait rien écrit dans `out/`). Le dossier n'était donc pas
créé, l'écriture de `console.log` échouait sans bruit et l'artefact était vide.
Le workflow crée désormais le dossier d'abord, garde toujours `status.txt`,
`ls.txt` et `console.log`, affiche les 60 dernières lignes de la sortie du job
dans le run, et publie l'artefact même si le téléchargement échoue. Retesté sur
un job terminé, un job sans fichiers et un id inconnu.

Second correctif : l'identifiant complet d'un job est `<clé d'équipe>--<id>`.
`habs watch` accepte l'id court, mais `status`, `ls`, `get` et `logs` exigent
l'id complet ; d'où « no jobs yet » et « wrote nothing » au lieu des vrais
résultats. Le workflow complète désormais l'id et liste tous les jobs de
l'équipe. Retesté sur le stockage simulé (id court et id complet).

## 2026-11-22 — Premier job GPU : bloqué à l'extraction, tué à 1 h

**Constat** (log complet récupéré par le workflow « HABS results ») : le job
`20261122T003121-0fcdc8` a démarré à 00:36 sur une H100 NVL (40 CPU,
`cuda: true`), a extrait **une seule séance en une heure**, puis a été tué
par la plateforme (`MaxWallClockTimeExceeded`, 1:00:57). Un job tué ne garde
rien : `out/` est vide. 1,0 h GPU consommée pour rien.

**Cause probable** : l'extraction lançait un pool de processus par *fork*
depuis un parent qui venait de charger torch et d'initialiser CUDA (pour
afficher le nom du GPU) ; chaque processus utilisait en plus tous les cœurs
pour ses calculs (40 × 40 threads). Le Mac extrait les 590 séances en 18 min
avec un seul processus : l'extraction n'était pas lente, elle était bloquée.

**Correctifs** (`reid/platform.py`) :
- chaque morceau d'extraction (cache features et cache brut, `workers`
  morceaux chacun) est un processus `scripts/reid_extract.py` indépendant, sans
  fork, avec un seul thread de calcul ; un log par morceau dans `out/logs/` ;
- les groupes de modèles se partagent les CPU (`OMP_NUM_THREADS` = CPU /
  groupes) ;
- limite interne `deadline_min` (52 min dans `reid_job.yaml`) : à l'échéance,
  ce qui tourne encore est arrêté, `job.json` est écrit quand même et les
  fichiers de résultats des modèles finis sont conservés ; les morceaux
  interrompus remontent ce qu'ils ont extrait.

**Tests** :
- 2 tests (`tests/reid/test_platform.py`) : lecture des compteurs, arrêt à
  l'échéance d'un processus qui dort, codes de sortie. Le premier a trouvé un
  bug du correctif lui-même (découpage de la ligne au mauvais `": "`) ;
- job complet en local sur les vraies données (18 CPU, caches vides) :
  limite de 1 min respectée (60,1 s, 358 séances extraites, rapport écrit) ;
  sans limite, extraction des deux caches en 105 s (590 + 590, aucun échec),
  puis deux groupes en parallèle : `fuse_ref_lda3_s0` 0,827 [0,774-0,880] et
  `fuse_lda3_resp_s0` 0,851 [0,803-0,899], chance 0,024, identiques aux runs
  précédents ; total 3 min 42 s ;
- suite complète : 54 tests passent, ruff propre.

**À surveiller au prochain job** : durée de l'entraînement des encodeurs sur
H100 (≈ 40 min par graine sur le Mac). Si les 3 graines ne tiennent pas dans
52 min, les modèles finis sont gardés et on répartit le reste sur un 2e job.

## 2026-11-22 — Découper le travail GPU en jobs de moins d'une heure

Un job ne reçoit aucun paramètre : il exécute le commit soumis. Pour lancer des
morceaux différents sans un commit sur main par job :
- `reid_job.yaml` décrit des **parts** (`all` : les 3 graines en 6 processus ;
  `s0`, `s1`, `s2` : une graine en 2 processus, fusions LDA d'un côté,
  fusions à encodeurs de l'autre) ; `part: all` par défaut ;
- `reid/platform.py` (`select_part`) exécute la part écrite dans
  `reid_job_part` si ce fichier existe, sinon `part` ;
- nouveau workflow `.github/workflows/reid_gpu.yml` (« reid GPU », hors reid/,
  signalé ; `submit.yml` intact) : champ `part`, écrit `reid_job_part` dans un
  commit fait sur la machine GitHub seulement (jamais poussé ; `habs submit`
  envoie `git archive HEAD`), soumet, suit le job puis joint ses résultats en
  artefact (sans les `failures_*.csv`). `reid_job_part` est ignoré par git.

Tests : 2 de plus (choix de la part par fichier ou par défaut, part inconnue
refusée ; tous les modèles cités dans les parts existent). L'étape du
workflow a été exécutée telle quelle sur des clones jetables : part valide
committée et présente dans l'archive, part inconnue et nom invalide refusés.
56 tests passent.
