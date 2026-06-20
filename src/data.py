from sklearn.datasets import load_breast_cancer
import pandas as pd


def load_data():
    """Built-in binary classification dataset: 569 rows, 30 features."""
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    return X, y


def sanity_check(X: pd.DataFrame, y: pd.Series) -> None:
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("\nDtypes:\n", X.dtypes.value_counts())
    print("\nClass balance (0=malignant, 1=benign):\n", y.value_counts(normalize=True))
