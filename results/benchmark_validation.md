# Benchmark sur la validation

Optimisation Optuna sur le train (CV stratifiée par groupes de profils), évaluation sur la validation. Le test n'est pas utilisé.

| Modèle | Origine | F1 macro CV (train) | F1 macro val. | Balanced acc. val. | Accuracy val. | F1 Performant | F1 Passable | F1 Energivore |
|---|---|---|---|---|---|---|---|---|
| Random Forest | historique | 0.748 | 0.775 | 0.794 | 0.812 | 0.745 | 0.857 | 0.722 |
| Gradient Boosting | historique | 0.750 | 0.773 | 0.792 | 0.820 | 0.720 | 0.864 | 0.734 |
| LightGBM | nouveau | 0.755 | 0.769 | 0.783 | 0.835 | 0.711 | 0.884 | 0.712 |
| CatBoost | nouveau | 0.753 | 0.767 | 0.814 | 0.817 | 0.708 | 0.863 | 0.729 |
| XGBoost | historique | 0.744 | 0.763 | 0.812 | 0.791 | 0.743 | 0.835 | 0.713 |
| Arbre de décision | historique | 0.719 | 0.751 | 0.746 | 0.815 | 0.715 | 0.871 | 0.668 |
| SVM (noyau RBF) | historique | 0.704 | 0.748 | 0.742 | 0.815 | 0.703 | 0.871 | 0.669 |
| AdaBoost | historique | 0.720 | 0.741 | 0.769 | 0.799 | 0.696 | 0.854 | 0.673 |
| Logistique Lasso | historique | 0.697 | 0.723 | 0.727 | 0.809 | 0.659 | 0.869 | 0.642 |
| Logistique Elastic Net | historique | 0.701 | 0.716 | 0.760 | 0.776 | 0.644 | 0.831 | 0.673 |
| Logistique Ridge | historique | 0.706 | 0.690 | 0.667 | 0.781 | 0.556 | 0.837 | 0.675 |
