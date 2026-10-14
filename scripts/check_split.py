
"""Vérification de la séparation Train / Validation / Test."""

from dpe.config import GROUPS
from dpe.data_loader import load_processed_data
from dpe.preprocessing import split_features_target
from dpe.training import (
    make_profile_groups,
    split_data,
    split_train_validation_test,
)


def display_distribution(name, X, y, total):
    """Affiche la taille et la répartition des classes."""

    print(f"\n=== {name} ===")
    print(f"Observations : {len(X)}")
    print(f"Proportion   : {100 * len(X) / total:.2f} %")

    print("\nRépartition des classes (%) :")

    proportions = (
        y.value_counts(normalize=True)
        .reindex(GROUPS, fill_value=0)
        .mul(100)
        .round(2)
    )

    print(proportions.to_string())


def main():
    # 1. Charger les données.
    df = load_processed_data()
    X, y = split_features_target(df)

    # 2. Construire les trois ensembles.
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_train_validation_test(X, y)

    total = len(df)

    print("\n=== DONNEES COMPLETES ===")
    print(f"Observations : {total}")
    print(f"Variables explicatives : {X.shape[1]}")

    # 3. Examiner chaque ensemble.
    display_distribution(
        "TRAIN", X_train, y_train, total
    )

    display_distribution(
        "VALIDATION", X_val, y_val, total
    )

    display_distribution(
        "TEST FINAL", X_test, y_test, total
    )

    # 4. Vérifier les profils communs.
    train_groups = set(make_profile_groups(X_train))
    val_groups = set(make_profile_groups(X_val))
    test_groups = set(make_profile_groups(X_test))

    overlap_train_val = len(train_groups & val_groups)
    overlap_train_test = len(train_groups & test_groups)
    overlap_val_test = len(val_groups & test_groups)

    print("\n=== CONTROLE DES PROFILS COMMUNS ===")
    print("Train / Validation :", overlap_train_val)
    print("Train / Test       :", overlap_train_test)
    print("Validation / Test  :", overlap_val_test)

    assert overlap_train_val == 0
    assert overlap_train_test == 0
    assert overlap_val_test == 0

    # 5. Vérifier que toutes les observations sont conservées.
    assert len(X_train) + len(X_val) + len(X_test) == total

    # 6. Vérifier la présence des trois classes.
    for name, labels in [
        ("TRAIN", y_train),
        ("VALIDATION", y_val),
        ("TEST", y_test),
    ]:
        missing_classes = set(GROUPS) - set(labels.unique())

        if missing_classes:
            raise RuntimeError(
                f"Classes absentes dans {name} : {missing_classes}"
            )

    # 7. Vérifier que le test final n'a pas changé.
    _, X_test_original, _, y_test_original = split_data(
        X,
        y,
        strategy="group",
    )

    assert X_test.index.equals(X_test_original.index)
    assert y_test.index.equals(y_test_original.index)

    print("\n=== RESULTAT ===")
    print("Aucun profil commun entre les trois ensembles.")
    print("Toutes les observations sont conservées.")
    print("Les trois classes sont présentes partout.")
    print("Le test final initial est préservé.")
    print("\nSéparation validée.")


if __name__ == "__main__":
    main()
