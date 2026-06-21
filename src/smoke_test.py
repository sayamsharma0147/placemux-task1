import sys

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

from .config import set_seed, SEED
from .data import load_data, sanity_check
from .split import make_splits
from .experiment_log import log_experiment


def main():
    set_seed()

    try:
        X, y = load_data()
        sanity_check(X, y)

        X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
        print(f"\nTrain: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

        model = DummyClassifier(strategy="most_frequent", random_state=SEED)
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds)

        print(f"\n[Smoke Test] Val Accuracy: {acc:.3f} | Val F1: {f1:.3f}")

        log_experiment(
            run_id="smoke_test_dummy",
            params={"model": "DummyClassifier", "strategy": "most_frequent", "seed": SEED},
            metrics={"val_accuracy": round(acc, 4), "val_f1": round(f1, 4)},
        )

    except (ValueError, RuntimeError) as e:
        print(f"\nSMOKE TEST FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nSMOKE TEST FAILED (unexpected error): {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()