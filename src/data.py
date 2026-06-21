from sklearn.datasets import load_breast_cancer
import pandas as pd


def load_data():
    """Built-in binary classification dataset: 569 rows, 30 features."""
    try:
        data = load_breast_cancer(as_frame=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}") from e

    X = data.data
    y = data.target

    if X is None or y is None or len(X) == 0:
        raise ValueError("Loaded dataset is empty.")

    return X, y


def sanity_check(X: pd.DataFrame, y: pd.Series) -> None:
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch: X has {X.shape[0]} rows, y has {y.shape[0]} rows.")

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("\nDtypes:\n", X.dtypes.value_counts())

    missing = X.isnull().sum().sum()
    print(f"\nMissing values in X: {missing}")
    if missing > 0:
        print("WARNING: missing values found. Decide on an imputation/drop strategy before modeling.")

    class_counts = y.value_counts()
    print("\nClass balance (0=malignant, 1=benign):\n", y.value_counts(normalize=True))

    if class_counts.shape[0] < 2:
        raise ValueError("Target has fewer than 2 classes -- cannot do stratified classification split.")

    rare_classes = class_counts[class_counts < 5]
    if not rare_classes.empty:
        print(f"WARNING: classes with very few samples (may break stratified splitting): \n{rare_classes}")