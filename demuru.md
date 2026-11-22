
Prévisualisation de README_DEMURU.md
# Demuru & Fraschini 2020 sur le dataset HABS

Reproduction de « EEG fingerprinting: subject-specific signature based on the aperiodic
component of power spectrum » (Computers in Biology and Medicine, 2020,
https://arxiv.org/abs/2001.09424) sur les enregistrements de repos du hackathon.

## Les fichiers

| Fichier | Rôle |
|---|---|
| `process.py` | Point d'entrée attendu par la plateforme HABS (remplace celui du template). |
| `demuru_fingerprint.py` | Toute la méthode. Les réglages sont dans `CONFIG`, en haut du fichier. |
| `requirements.txt` | Dépendances (celles du template + fooof, scikit-learn, matplotlib, joblib). |

Copiez ces trois fichiers à la racine du dépôt de votre équipe (celui créé à partir du
template), en remplaçant `process.py` et `requirements.txt`.

## Ce que fait le script

1. Cherche `PREPROCESSED/P*/P*_day-*_task-resting-state_multimodal.bdf`.
2. Charge seulement les 62 électrodes EEG (d'après le `.record_map`) et retire le
   remplissage final (on garde `TotalSamples` points).
3. Les fichiers PREPROCESSED sont déjà nettoyés par l'équipe HABS : filtre 0,5-48 Hz,
   référence moyenne, ICA (Picard) + ICLabel. C'est l'équivalent de l'étape ICA de l'article.
   Le script rééchantillonne simplement à 250 Hz.
4. Repère les conditions grâce aux marqueurs du fichier :
   - yeux ouverts : marqueur 10002 (ou 10000), environ 30 s → 2 fenêtres de 12 s ;
   - yeux fermés : marqueur 10004, 60 s → 5 fenêtres de 12 s, exactement comme l'article.
5. Rejette une fenêtre si une électrode dépasse 300 µV crête à crête ou si FOOOF ajuste mal
   (R² médian < 0,80).
6. Pour chaque fenêtre et chaque électrode : spectre de Welch (fenêtres de 2 s, 50 % de
   recouvrement), FOOOF entre 1 et 45 Hz → offset et exposant ; puissance relative
   thêta, alpha, bêta, gamma (bandes de l'article).
7. Compare les fenêtres : similarité = 1 / (1 + distance euclidienne), comme l'article.
8. Calcule, pour chaque condition et chaque jeu de features :
   - **Test A, même jour** (comme l'article) : EER et identification au rang 1 ;
Demuru & Fraschini 2020 sur le dataset HABS
Reproduction de « EEG fingerprinting: subject-specific signature based on the aperiodic component of power spectrum » (Computers in Biology and Medicine, 2020, https://arxiv.org/abs/2001.09424) sur les enregistrements de repos du hackathon.

Les fichiers
Fichier	Rôle
process.py	Point d'entrée attendu par la plateforme HABS (remplace celui du template).
demuru_fingerprint.py	Toute la méthode. Les réglages sont dans CONFIG, en haut du fichier.
requirements.txt	Dépendances (celles du template + fooof, scikit-learn, matplotlib, joblib).
Copiez ces trois fichiers à la racine du dépôt de votre équipe (celui créé à partir du template), en remplaçant process.py et requirements.txt.

Ce que fait le script
Cherche PREPROCESSED/P*/P*_day-*_task-resting-state_multimodal.bdf.
Charge seulement les 62 électrodes EEG (d'après le .record_map) et retire le remplissage final (on garde TotalSamples points).
Les fichiers PREPROCESSED sont déjà nettoyés par l'équipe HABS : filtre 0,5-48 Hz, référence moyenne, ICA (Picard) + ICLabel. C'est l'équivalent de l'étape ICA de l'article. Le script rééchantillonne simplement à 250 Hz.
Repère les conditions grâce aux marqueurs du fichier :
yeux ouverts : marqueur 10002 (ou 10000), environ 30 s → 2 fenêtres de 12 s ;
yeux fermés : marqueur 10004, 60 s → 5 fenêtres de 12 s, exactement comme l'article.
Rejette une fenêtre si une électrode dépasse 300 µV crête à crête ou si FOOOF ajuste mal (R² médian < 0,80).
Pour chaque fenêtre et chaque électrode : spectre de Welch (fenêtres de 2 s, 50 % de recouvrement), FOOOF entre 1 et 45 Hz → offset et exposant ; puissance relative thêta, alpha, bêta, gamma (bandes de l'article).
Compare les fenêtres : similarité = 1 / (1 + distance euclidienne), comme l'article.
Calcule, pour chaque condition et chaque jeu de features :
Test A, même jour (comme l'article) : EER et identification au rang 1 ;
Test B, jour 1 → jours suivants (notre question) : EER et identification au rang 1, seulement pour les participants qui ont au moins deux jours.
Lancer en local (terminal de VS Code)
bash
Copier
# 1. Environnement (une fois)
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt

# 2. Test sur l'échantillon fourni dans le dépôt (1 participant : pas d'évaluation possible)
python process.py --data ./sample --out ./out_sample

# 3. Télécharger le repos de tout le monde (environ 20 Mo par fichier)
pip install huggingface_hub
hf auth login
hf download HABSteam/multimodal-eeg-emotion --repo-type dataset --local-dir ./data \
    --include "PREPROCESSED/*/*resting-state*"

# 4. Test rapide sur 30 participants, puis tout le monde
python process.py --data ./data --out ./out_30 --max-participants 30
python process.py --data ./data --out ./out_tous
Avec la plateforme : habs run --data ./data --out ./out appelle le même process().

Lancer sur la plateforme HABS
Le dépôt de l'équipe doit avoir la clé d'équipe dans .github/workflows/submit.yml.
Faites un commit et un push des trois fichiers.
Sur GitHub : Actions → HABS → Run workflow.
Les résultats arrivent dans le dossier out/ du job (lien de résultats de l'équipe).
Rien à télécharger : tout le dataset est déjà monté dans /data. Le calcul n'utilise pas le GPU (tout se fait sur le processeur, en parallèle sur tous les cœurs). L'intérêt de la plateforme ici est d'avoir toutes les données sans téléchargement.

Durée indicative : environ 8 s par fichier sur un cœur, donc de l'ordre de 10 à 15 min pour quelques centaines de fichiers sur 8 cœurs.

Ce qui sort
Fichier	Contenu
resume.txt	Le tableau des résultats, lisible directement.
resultats.csv	Une ligne par condition × test × jeu de features : EER, AUC, rang 1, hasard, EER de l'article.
features_fenetres.csv.gz	Une ligne par fenêtre : participant, jour, condition, puis offset__Fp1, exponent__Fp1, rel_alpha__O1…
journal_fichiers.csv	Chaque fichier : traité ou en erreur, fenêtres gardées ou rejetées, et pourquoi.
figures/eer_par_features_EC.png	EER de l'article, le nôtre même jour, et le nôtre d'un jour à l'autre.
figures/scores_offset_EC.png	Distribution des similarités « même personne » contre « personnes différentes ».
Les mêmes figures existent pour EO (yeux ouverts).

Comment lire les résultats
eer : taux d'erreur en vérification, au réglage où faux acceptés = faux rejets. Plus bas = mieux. L'article trouve 0,057 avec l'offset, yeux fermés.
rank1 : part des fenêtres attribuées à la bonne personne parmi toutes. À comparer à chance_rank1 (le hasard, 1 / nombre de participants).
Test A contre test B : c'est la chute entre les deux qui dit si la signature tient d'un jour à l'autre. A_meme_jour_sous_groupe utilise les mêmes participants que le test B, pour une comparaison équitable.
Différences avec l'article, à garder en tête
62 électrodes à 1 000 Hz (ramenées à 250 Hz) au lieu de 64 à 160 Hz.
ICA : Picard + ICLabel (fait par HABS) au lieu d'ADJUST.
Yeux ouverts : 30 s seulement (consigne « sans cligner »), donc 2 fenêtres au lieu de 5.
Les paramètres exacts de Welch et de FOOOF ne sont pas donnés dans l'article : ceux de CONFIG sont des choix standard.
Le nombre de participants diffère : les EER se comparent en ordre de grandeur.
Réglages utiles dans CONFIG
max_participants : limiter pour un test.
data_version : "CURATED" pour partir des données brutes (filtrage refait, sans ICA).
reject_peak_to_peak_uv, min_median_r2 : sévérité du contrôle qualité.
fooof_settings, welch_segment_s : réglages du spectre et de FOOOF.
