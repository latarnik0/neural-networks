"""
Multiclass MLP classification: Iris and MNIST.

Trains a multilayer perceptron (sklearn MLPClassifier) with one hidden layer,
softmax output, and one-hot target encoding. Hidden activations: tanh / ReLU.
Solvers: adam, sgd, lbfgs.

The task is split into two parts:

Part 1 — Iris: 3 species from 4 flower measurements.
Part 2 — MNIST: handwritten digits 0-9; each 28x28 image flattened to a
784-dim input vector.

Hyperparameters are selected with GridSearchCV over:
  - hidden_layer_sizes: (20,), (40,), …, (100,)
  - learning_rate_init: 0.1, 0.01, 0.001
  - activation: tanh, relu
  - solver: adam, lbfgs, sgd

Results for both datasets are reported with confusion matrices.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[2]
IRIS_PATH = ROOT / "iris_data.txt"
PLOTS_DIR = ROOT / "plots" / "task2"

# Hyperparameter grid
PARAM_GRID = {
    "mlp__hidden_layer_sizes": [(n,) for n in range(20, 101, 20)],
    "mlp__learning_rate_init": [0.1, 0.01, 0.001],
    "mlp__activation": ["tanh", "relu"],
    "mlp__solver": ["adam", "lbfgs", "sgd"],
}

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


def dataset_to_xy(dataset):
    """Convert {features: label} dict to X (n, d) and y (n,) arrays."""
    X = np.array([list(features) for features in dataset.keys()], dtype=float)
    y = np.array(list(dataset.values()))

    return X, y


def load_iris_xy(filename=IRIS_PATH):
    """Load Iris as separate train / validation / test sets (70% / 15% / 15%)."""
    train, val, test = read_iris_data_3_classes(filename)
    X_train, y_train = dataset_to_xy(train)
    X_val, y_val = dataset_to_xy(val)
    X_test, y_test = dataset_to_xy(test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_mnist_xy(val_size=0.15, random_state=42):
    """
    Load MNIST data and split into train / val / test.

    Official split: 60k train + 10k test. From the 60k we carve out a
    stratified validation set (default 15% ≈ 9k), leaving ~51k for training.
    """

    print("Loading MNIST via sklearn fetch_openml (may download once)...")
    mnist = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="liac-arff"
    )
    X = np.asarray(mnist.data, dtype=float)
    y = np.asarray(mnist.target, dtype=int)
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]
    print("MNIST loaded via OpenML")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
