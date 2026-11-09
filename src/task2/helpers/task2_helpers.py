from pathlib import Path

import numpy as np

# neural_networks/ and task2/
ROOT = Path(__file__).resolve().parents[3]
TASK2_DIR = Path(__file__).resolve().parents[1]
IRIS_PATH = ROOT / "src" / "task1" / "iris_data.txt"
PLOTS_DIR = TASK2_DIR / "plots"

# Hyperparameter grid from the lab PDF
PARAM_GRID = {
    "mlp__hidden_layer_sizes": [(n,) for n in range(20, 101, 20)],
    "mlp__learning_rate_init": [0.1, 0.01, 0.001],
    "mlp__activation": ["tanh", "relu"],
    "mlp__solver": ["adam", "lbfgs", "sgd"],
}

IRIS_MAX_ITER = 500


def dataset_to_xy(dataset):
    """Convert {features: label} dict to X (n, d) and y (n,) arrays."""
    X = np.array([list(features) for features in dataset.keys()], dtype=float)
    y = np.array(list(dataset.values()))
    return X, y
