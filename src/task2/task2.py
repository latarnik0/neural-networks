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

from helpers import *

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
    """Convert raw loaded data to fit the MLP"""

    train, val, test = read_iris_data_3_classes(filename)
    X_train, y_train = dataset_to_xy(train)
    X_val, y_val = dataset_to_xy(val)
    X_test, y_test = dataset_to_xy(test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_mnist_xy(val_size=0.15, random_state=42):
    """Get MNIST data using fetch_openml"""

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
