from sklearn.model_selection import train_test_split
from .config import SEED


def make_splits(X, y, test_size=0.2, val_size=0.25):
    """
    60/20/20 train/val/test split, stratified so class balance
    is preserved in every slice.

    test_size: fraction carved out as the test set.
    val_size: fraction of the REMAINING train+val pool used for validation.
              (0.25 of the 80% leftover = 20% of the total)
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, stratify=y_trainval, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
