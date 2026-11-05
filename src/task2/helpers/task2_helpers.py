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


def dataset_to_xy(dataset):
    """Convert {features: label} dict to X (n, d) and y (n,) arrays."""
    X = np.array([list(features) for features in dataset.keys()], dtype=float)
    y = np.array(list(dataset.values()))

    return X, y