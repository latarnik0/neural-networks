"""
Multiclass MLP classification: Iris and MNIST.

Trains a MLPClassifier with one hidden layer, softmax output, and one-hot target encoding.
Hidden activations: tanh / ReLU.
Solvers: adam, sgd, lbfgs.

The task is split into two parts:

Part 1 — Iris: 3 species from 4 flower measurements.
Part 2 — MNIST: handwritten digits 0-9; each 28x28 image flattened to a
784-dim input vector.

Hyperparameters are selected with GridSearchCV:
  - hidden_layer_sizes: (20,), (40,), …, (100,)
  - learning_rate_init: 0.1, 0.01, 0.001
  - activation: tanh, relu
  - solver: adam, lbfgs, sgd

Results for both datasets are reported with confusion matrices.
"""

import warnings

from helpers.task2_helpers import *

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_iris_data_3_classes(filename, train_limit_per_class=35, val_limit_per_class=42):
    """Read Iris file into train / validation / test dicts (70% / 15% / 15%)."""
    learning_dataset = {}
    validation_dataset = {}
    test_dataset = {}
    counters = {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0}

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            entry_list = line.split(",")
            features = tuple(map(float, entry_list[:4]))
            species = entry_list[-1]

            if species in counters:
                counters[species] += 1
                if counters[species] <= train_limit_per_class:
                    learning_dataset[features] = species
                elif counters[species] <= val_limit_per_class:
                    validation_dataset[features] = species
                else:
                    test_dataset[features] = species

    return learning_dataset, validation_dataset, test_dataset


def load_iris_xy(filename=IRIS_PATH):
    """Load Iris as separate train / validation / test sets (70% / 15% / 15%)."""
    train, val, test = read_iris_data_3_classes(filename)
    X_train, y_train = dataset_to_xy(train)
    X_val, y_val = dataset_to_xy(val)
    X_test, y_test = dataset_to_xy(test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_mnist_xy(val_size=0.15, random_state=42):
    """Get MNIST data using fetch_openml (not used yet — Part 2)."""
    from sklearn.datasets import fetch_openml

    print("Loading MNIST via sklearn fetch_openml...")
    mnist = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="liac-arff"
    )
    X = np.asarray(mnist.data, dtype=float)
    y = np.asarray(mnist.target, dtype=int)
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]
    print("MNIST loaded")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# MLP + GridSearchCV
# ---------------------------------------------------------------------------

def make_mlp_pipeline(max_iter):
    """StandardScaler + MLPClassifier (multiclass → softmax output)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    max_iter=max_iter,
                    random_state=42,
                ),
            ),
        ]
    )


def run_grid_search(X, y, max_iter, cv=3, n_jobs=-1):
    """Exhaustive hyperparameter search from the lab grid."""
    grid = GridSearchCV(
        estimator=make_mlp_pipeline(max_iter),
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )
    grid.fit(X, y)
    return grid


def print_best_params(grid, dataset_name):
    print(f"\n=== {dataset_name}: best hyperparameters ===")
    for key, value in grid.best_params_.items():
        print(f"  {key}: {value}")
    print(f"  Best CV accuracy: {grid.best_score_ * 100:.2f}%")
    mlp = grid.best_estimator_.named_steps["mlp"]
    print(f"  Output activation: {mlp.out_activation_}")


def evaluate_and_plot(model, X_test, y_test, title, save_path, labels=None):
    """Predict on test set, print accuracy, save confusion matrix."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"{title}")
    print(f"  Test accuracy: {acc * 100:.2f}%")
    print(f"  Confusion matrix:\n{cm}\n")

    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return acc, cm


# ---------------------------------------------------------------------------
# Main — Iris only
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("--- Part 1: Iris (MLP + GridSearchCV) ---")
    X_train, y_train, X_val, y_val, X_test, y_test = load_iris_xy()
    iris_labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    print(
        f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    # GridSearchCV uses CV on the training set; val is held out for a check
    iris_grid = run_grid_search(X_train, y_train, max_iter=IRIS_MAX_ITER, cv=3)
    print_best_params(iris_grid, "Iris")

    best = iris_grid.best_estimator_
    val_acc = accuracy_score(y_val, best.predict(X_val))
    print(f"  Validation accuracy: {val_acc * 100:.2f}%")

    evaluate_and_plot(
        best,
        X_test,
        y_test,
        title="Iris — MLP confusion matrix",
        save_path=PLOTS_DIR / "iris_confusion_matrix.png",
        labels=iris_labels,
    )

    print("\nDone (Iris only).")
