# Audit de l'existant et plan de migration

Date de l'audit : 21/11/2026. Dernier commit au moment de l'audit : `1457104`.

## 1. État du dépôt et de l'environnement

| Élément | Constat |
|---|---|
| Branche | `main`, à jour avec `origin/main` |
| Fichiers non suivis | `notebooks/04_svm_audit.txt` (export texte des 52 cellules du notebook SVM) |
| Tests | 19 tests pytest réussis (`python -m pytest -q`) |
| Données | `data/processed/data_for_processing_final.xlsx` : 55 357 lignes, 19 variables + cible, aucune valeur manquante (non versionné, cf. `.gitignore`) |
| Fins de ligne | dépôt en LF, copie de travail Windows en CRLF (`core.autocrlf=true`) |
| Environnement local | `.venv` Windows, Python 3.14.3 |

Les expériences de cette migration ont été exécutées sous Linux (Python 3.13,
scikit-learn 1.9.1, pandas 3.0, Optuna 5.1, XGBoost 3.4, LightGBM 4.7,
CatBoost 1.2.10). Les versions sont figées dans `requirements.txt`.

### Vérification de la séparation Train / Validation / Test

`scripts/check_split.py` a été relancé : la séparation par empreintes de
profils est **correcte et conservée telle quelle**.

| Ensemble | Observations | Performant | Passable | Energivore |
|---|---|---|---|---|
| Train | 33 351 | 4,2 % | 72,1 % | 23,7 % |
| Validation | 11 130 | 3,5 % | 70,3 % | 26,2 % |
| Test | 10 876 | 4,8 % | 70,7 % | 24,5 % |

Aucun profil commun entre les trois ensembles ; toutes les observations sont
conservées ; le test final est identique à celui de la baseline historique.

## 2. Pourquoi la séparation par profils est indispensable

| Indicateur | Valeur |
|---|---|
| Profils de variables distincts | 34 970 |
| Lignes appartenant à un profil dupliqué | 24 936 (45,0 %) |
| Profils portant des étiquettes contradictoires | 908 (5 955 lignes) |

Près de la moitié des lignes partagent exactement leurs 19 variables avec une
autre ligne. Avec le `train_test_split` stratifié utilisé dans **tous** les
notebooks historiques, un même profil se retrouve à la fois en train et en
test : les modèles capables de mémoriser (arbres profonds, forêts, boosting)
sont alors favorisés artificiellement. Les scores historiques ne sont donc pas
directement comparables aux scores obtenus avec le protocole actuel.

Les 908 profils contradictoires (même description, étiquettes différentes)
fixent un plafond de performance : une partie de l'erreur est irréductible
avec ces 19 variables.

## 3. Audit des fuites de cible

Script : `scripts/audit_leakage.py` → `results/metrics/leakage_audit.csv`.

- Aucune colonne ne correspond à une sortie du calcul DPE (consommation,
  émissions GES, étiquette GES, coût, besoins, déperditions).
- Meilleure variable seule (règle « modalité → classe », validation) :
  `type_generateur_n1_ecs_n1`, F1 macro 0,48. Aucune variable n'approche le
  seuil d'alerte de 0,90.
- Les variables sont des **entrées** du calcul réglementaire 3CL (isolation,
  générateurs, période de construction, ventilation). Elles sont légitimement
  disponibles avant le calcul de l'étiquette : le modèle apprend à approcher
  la méthode réglementaire à partir d'une description partielle du logement.

Point de vigilance (non bloquant) : le prétraitement amont (notebook 02) a
construit les regroupements de modalités à partir de tableaux croisés avec la
cible sur **l'ensemble** des données (dendrogrammes, modalités < 1 %). C'est
une légère fuite d'information de conception, figée dans le fichier
`data_for_processing_final.xlsx`. Elle est documentée ici et n'a pas été
corrigée, car cela impose de repartir des données brutes (non présentes dans
le dépôt).

## 4. Inventaire des notebooks historiques

Protocole commun à tous les notebooks 03 à 08 : `train_test_split` stratifié
80/20 (`random_state=42`), `StratifiedKFold(5)` pour Optuna, sélection et
comparaison **sur le jeu de test**.

Prétraitement commun des modèles (notebooks 03 à 08) :
`ColumnTransformer` = `RobustScaler` (variables continues ; `passthrough`
pour les arbres) + `category_encoders.TargetEncoder(smoothing=3,
min_samples_leaf=50)` sur les 10 variables qualitatives + `passthrough` pour
les 6 variables ordinales `_enc` et la variable binaire. Le `TargetEncoder`
de category_encoders code la cible textuelle par ordre alphabétique
(Energivore = 0, Passable = 1, Performant = 2) : chaque modalité est remplacée
par la moyenne lissée de ce score ordinal.

Déséquilibre : poids de classe optimisés par Optuna
(`Performant` ∈ [1, 25], `Energivore` ∈ [1, 15], `Passable` = 1).

### 01_eda.ipynb
Données brutes ADEME (57 826 lignes, 46 colonnes, étiquettes A à G).
Analyse des manquants (test du χ² : manquants non aléatoires), outliers
(`hauteur_sous_plafond` jusqu'à 240 m), asymétrie, V de Cramér.

### 02_preprocessing.ipynb
Regroupement A-B / C-D / E-F-G, suppression des identifiants et des colonnes
quasi constantes (> 95 %), écrêtage de `hauteur_sous_plafond` à 6 m,
`log1p` sur `volume_stockage_generateur_n1_ecs_n1`, suppression des colonnes
> 45 % de manquants puis des lignes incomplètes (57 826 → 55 357 lignes),
suppression de 4 variables redondantes, agrégation de modalités (dendrogrammes),
modalités < 1 % → `Autre`, encodage ordinal de 6 variables. Export :
55 357 × 20.

### 03_logistic_regression.ipynb
Trois variantes, 25 essais Optuna chacune :

| Variante | Espace de recherche | Meilleurs paramètres | F1 CV | F1 test |
|---|---|---|---|---|
| Elastic Net (saga) | C ∈ [1e-3, 100] log, l1_ratio ∈ [0, 1] | C = 3,35, l1 = 0,80, poids 2/1/2 | 0,716 | 0,726 |
| Ridge (lbfgs) | C ∈ [1e-3, 100] log | poids 1/1/3 | 0,700 | 0,704 |
| Lasso (saga) | C ∈ [1e-2, 10] log | poids 1/1/3 | 0,706 | 0,708 |

### 04_svm.ipynb
- Sous-échantillon stratifié de 8 000 lignes du train pour toutes les étapes
  coûteuses.
- Criblage des noyaux (C = 1, `class_weight='balanced'`) : linéaire 0,618,
  polynomial (degré 3) 0,671, RBF 0,653.
- Optuna (25 essais) sur le noyau RBF : C ∈ [1e-2, 100] log,
  gamma ∈ [1e-4, 10] log → C = 34,5, gamma = 0,242, poids 1/1/13,
  F1 CV 0,719.
- Modèle final entraîné sur les 8 000 lignes seulement, `probability=True`.
- Diagnostics : vecteurs de support, courbes de validation C et gamma,
  heatmap C × gamma, frontière de décision dans le plan des 2 premières
  composantes principales (ACP utilisée uniquement pour la visualisation,
  58,5 % de variance).

### 05_tree_random_forest.ipynb
| Modèle | Espace de recherche | Meilleurs paramètres | F1 CV | F1 test |
|---|---|---|---|---|
| Decision Tree | max_depth 3-30, min_samples_leaf 1-100, min_samples_split 2-100 | 30 / 20 / 71, poids 2/1/1 | 0,752 | 0,766 |
| Random Forest | n_estimators 100-500 (pas 50), max_depth 5-30, max_features 0,1-1, min_samples_leaf 1-50 | 200 / 27 / 0,80 / 2, poids 18/1/6 | 0,772 | 0,788 |

Diagnostics : arbre (3 niveaux), score OOB, effet de `n_estimators`,
importances Gini, courbe de validation sur `max_depth`.

### 06_boosting.ipynb
Poids passés par `sample_weight`. Pas de LightGBM ni de CatBoost.

| Modèle | Espace de recherche | Meilleurs paramètres | F1 CV | F1 test |
|---|---|---|---|---|
| AdaBoost | n_estimators 50-500, learning_rate 0,01-2 log | 300 / 0,41, poids 3/1/1 | 0,698 | 0,706 |
| Gradient Boosting | n_estimators 50-400, lr 0,01-0,5, max_depth 2-8, subsample 0,5-1 | 400 / 0,116 / 7 / 0,70, poids 6/1/5 | 0,773 | 0,780 |
| XGBoost | idem + reg_lambda 1e-3-10 log | 100 / 0,246 / 8 / 0,84 / 0,025, poids 2/1/7 | 0,775 | 0,777 |

### 07_model_comparison.ipynb
Réoptimisation (50 essais) d'Elastic Net, SVM RBF (8 000 lignes), Random
Forest et Gradient Boosting avec un préprocesseur unique (`RobustScaler`
partout). Classement **sur le test** : Gradient Boosting 0,799, Random Forest
0,790, SVM 0,743, Elastic Net 0,715.

### 08_tsne.ipynb
t-SNE 2D et 3D (perplexité 30, init PCA) sur 6 000 lignes du train, coloré
par la vraie classe et par la prédiction d'un Gradient Boosting (paramètres
du notebook 07, poids approximés 8/1/5), visualisations Plotly, zooms par
paire de classes.

## 5. Erreurs méthodologiques relevées

1. **Fuite par profils dupliqués** : split stratifié ligne à ligne alors que
   45 % des lignes appartiennent à des profils dupliqués (cf. section 2).
2. **Sélection sur le test** : dans les notebooks 03, 05, 06 et 07, le
   meilleur modèle est choisi d'après le score test. Le test sert donc de
   validation et son score est optimiste.
3. **Rapports de classification mal étiquetés** : `classification_report(...,
   target_names=GROUPS)` sans `labels=GROUPS`. scikit-learn trie les classes
   par ordre alphabétique (Energivore, Passable, Performant), donc les lignes
   « Performant » et « Energivore » sont inversées (support 2 699 attribué à
   Performant alors que c'est celui d'Energivore). Les interprétations par
   classe des notebooks 03, 04 et 06 sont à relire en conséquence.
4. **Double `log1p`** : `volume_stockage_generateur_n1_ecs_n1` est déjà
   transformé dans le fichier exporté par le notebook 02, mais les notebooks
   06, 07 et 08 réappliquent `log1p`.
5. **Versions de données hétérogènes** : les notebooks 03, 06, 07 et 08 ont
   été exécutés sur une version antérieure du fichier (25 colonnes, 9 valeurs
   manquantes, `configuration_installation_chauffage_n1` supprimée car
   quasi constante), les notebooks 04 et 05 sur la version actuelle
   (19 variables).
6. **SVM** : le noyau polynomial est meilleur au criblage mais c'est le RBF qui
   est optimisé (cellule « noyau retenu » laissée à compléter) ; le modèle final
   n'utilise que 8 000 lignes ; le nombre de vecteurs de support est rapporté
   à la taille du train complet (7,9 %) alors que le modèle n'a vu que 8 000
   lignes ; la conclusion annonce un F1 test de 0,743 alors que la sortie du
   notebook indique 0,707.
7. **t-SNE** : le `TargetEncoder` est ajusté sur les points projetés eux-mêmes
   et les erreurs affichées sont des erreurs sur des données d'entraînement.
8. **TargetEncoder sans validation croisée interne** : les statistiques de la
   cible sont calculées sur les lignes mêmes qu'elles encodent (biais optimiste
   modéré, limité par le lissage `min_samples_leaf=50`).
9. Interprétations « à compléter » laissées dans les notebooks 04, 05 et 07.

## 6. Différences entre les notebooks et le code Python existant

| Aspect | Notebooks | Code `src/dpe` (avant migration) |
|---|---|---|
| Split | stratifié ligne à ligne, train/test | par profils, train/validation/test |
| Encodage qualitatif | TargetEncoder (category_encoders) | OneHotEncoder |
| Mise à l'échelle | RobustScaler sur 2 variables continues | RobustScaler sur toutes les numériques |
| Régression logistique | Elastic Net, Ridge, Lasso + poids Optuna | L2 uniquement, grille sur C, `balanced` |
| Optimisation | Optuna, StratifiedKFold(5) | grille simple sur la validation |
| Modèles | LR, SVM, DT, RF, AdaBoost, GB, XGBoost | LR uniquement |

## 7. Plan de migration

Principes : un seul protocole pour tous les modèles ; Optuna sur le **train**
uniquement avec `StratifiedGroupKFold(5)` (groupes = empreintes de profils) ;
prétraitement ajusté dans chaque pli ; mêmes espaces de recherche que les
notebooks ; sélection sur la **validation** ; le test n'est utilisé qu'une
fois, pour le modèle choisi à l'avance.

| Étape | Contenu | Livrables |
|---|---|---|
| Socle | encodeur cible historique, registre des modèles, optimisation Optuna par groupes, benchmark validation | `models.py`, `optimization.py`, `benchmarking.py`, tests |
| SVM | criblage des 3 noyaux + Optuna RBF (8 000 lignes) + réentraînement sur le train complet | `results/metrics/svm_*` |
| Linéaires et arbres | Elastic Net, Ridge, Lasso ; Decision Tree ; Random Forest | `results/metrics/*` |
| Boosting | AdaBoost, Gradient Boosting, XGBoost (historiques) ; LightGBM, CatBoost (nouveaux) | `results/metrics/*` |
| Comparaison | tableau, métriques par classe, matrices de confusion, graphiques, analyse d'erreurs, rejeu du protocole historique | `compare_models.py` |
| Visualisation | ACP et t-SNE ajustés sur le train, projetés sur la validation | `visualize_projections.py` |
| Évaluation finale | un seul passage sur le test du modèle choisi sur la validation | `evaluate_final.py` |
| Documentation | README, instructions de reproduction | `README.md` |

Les expériences historiques (espaces de recherche des notebooks) et les
nouvelles expériences (LightGBM, CatBoost) sont distinguées par le champ
`origin` de chaque modèle (`historical` / `new`).
