"""
Single-neuron Iris classifiers: Perceptron and Adaline (binary and 3-class).

Implements Perceptron and Adaline with a Heaviside activation for Iris species
classification from four flower measurements. Binary mode separates setosa
from versicolor (labels -1 / +1). Three-class mode uses One-vs-Rest with three
binary Adaline units (setosa, versicolor, virginica).

Data is split 70% / 15% / 15% into train, validation, and test sets.
Training uses early stopping on the validation set (not train-error == 0):
after each epoch validation misclassifications are checked; training stops when
validation no longer improves for EARLY_STOPPING_PATIENCE epochs, then best
weights are restored.
Three-class accuracy stays near ~66.7% (2/3 species): versicolor and virginica
overlap strongly in feature space, so a single linear neuron cannot separate
them reliably; setosa remains linearly separable.

"""

import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMBER_OF_IRIS_PARAMETERS = 4
LEARNING_FACTOR = 0.001
NUMBER_OF_EPOCHS = 100
# Stop if validation error does not improve for this many epochs
EARLY_STOPPING_PATIENCE = 10

# Classic 2D view from Iris tutorials (sepal length, petal length)
VIZ_FEATURE_INDICES = (0, 2)
VIZ_FEATURE_LABELS = ("sepal length [cm]", "petal length [cm]")
FEATURE_NAMES = (
    "sepal length [cm]",
    "sepal width [cm]",
    "petal length [cm]",
    "petal width [cm]",
)


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

# Count class occurrences in the data file
def count_classes(filename):
    iris_setosa = 0
    iris_versicolor = 0

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            entry_list = line.split(",")

            if entry_list[-1] == "Iris-setosa":
                iris_setosa += 1
            elif entry_list[-1] == "Iris-versicolor":
                iris_versicolor += 1

    return iris_setosa, iris_versicolor


# Read Iris data — returns train / validation / test sets (70% / 15% / 15%)
def read_iris_data(filename, setosa_num_of_records, versicolor_num_of_records):
    learning_dataset = {}
    validation_dataset = {}
    test_dataset = {}

    iris_setosa_counter = 0
    iris_versicolor_counter = 0

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            entry_list = line.split(",")

            features = tuple(map(float, entry_list[:4]))
            species = entry_list[-1]

            if species == "Iris-setosa":
                label = -1.0
                iris_setosa_counter += 1

                if iris_setosa_counter <= 0.7 * setosa_num_of_records:
                    learning_dataset[features] = label
                elif iris_setosa_counter <= 0.85 * setosa_num_of_records:
                    validation_dataset[features] = label
                else:
                    test_dataset[features] = label

            elif species == "Iris-versicolor":
                label = 1.0
                iris_versicolor_counter += 1

                if iris_versicolor_counter <= 0.7 * versicolor_num_of_records:
                    learning_dataset[features] = label
                elif iris_versicolor_counter <= 0.85 * versicolor_num_of_records:
                    validation_dataset[features] = label
                else:
                    test_dataset[features] = label

    return learning_dataset, validation_dataset, test_dataset


# Read Iris data for 3-class classification (70% / 15% / 15%)
def read_iris_data_3_classes(filename, train_limit_per_class=35, val_limit_per_class=42):
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


def subset_features(dataset, feature_indices=VIZ_FEATURE_INDICES):
    """Keep only selected feature dimensions (for 2D decision-region plots)."""
    return {
        tuple(np.asarray(features)[list(feature_indices)]): label
        for features, label in dataset.items()
    }


def dataset_to_xy(dataset):
    """Convert {features: label} dict to X (n, d) and y (n,) arrays."""
    X = np.array([list(features) for features in dataset.keys()], dtype=float)
    y = np.array(list(dataset.values()))
    return X, y


# ---------------------------------------------------------------------------
# Training helpers (early stopping)
# ---------------------------------------------------------------------------

def count_misclassifications(model, dataset):
    """Count how many samples the model predicts incorrectly."""
    errors = 0
    for features, label in dataset.items():
        if model.predict(np.array(features)) != label:
            errors += 1
    return errors


def _early_stop_on_validation(
    model, validation_data, best_state, patience, epochs_without_improvement, epoch
):
    """
    After each epoch: check validation errors (early stopping).
    Returns (best_state, epochs_without_improvement, should_stop).
    best_state = (best_val_errors, best_weights, best_bias).

    Stop when validation does not improve for `patience` epochs.
    On a tie (same val score), keep the newer weights.
    """
    if validation_data is None:
        return best_state, epochs_without_improvement, False

    val_errors = count_misclassifications(model, validation_data)
    model.val_errors_.append(val_errors)

    best_val_errors, _, _ = best_state
    if val_errors < best_val_errors:
        # Validation improved — save weights and reset patience
        best_state = (val_errors, model.weights.copy(), model.bias)
        epochs_without_improvement = 0
    else:
        # Tie or worse: on a tie update to newer weights;
        # in both cases move toward early stop
        if val_errors == best_val_errors:
            best_state = (val_errors, model.weights.copy(), model.bias)
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(
                f"Early stop at epoch {epoch + 1}: "
                f"no validation improvement for {patience} epochs "
                f"(best val errors = {best_state[0]})."
            )
            return best_state, epochs_without_improvement, True

    return best_state, epochs_without_improvement, False


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Perceptron
class Perceptron:
    def __init__(self, learning_rate, epochs, n_features=NUMBER_OF_IRIS_PARAMETERS):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.uniform(0, 1, n_features)
        self.bias = random.uniform(0, 1)
        self.errors_ = []
        self.val_errors_ = []

    # Heaviside step activation
    def activation_function(self, vector_sum_of_neuron):
        if vector_sum_of_neuron >= 0.0:
            return 1.0
        else:
            return -1.0

    # Prediction — neuron output
    def predict(self, dataset_inputs):
        vector_sum = np.dot(dataset_inputs, self.weights) + self.bias
        return self.activation_function(vector_sum)

    # Training with early stopping on the validation set
    def train(
        self,
        training_data,
        validation_data=None,
        patience=EARLY_STOPPING_PATIENCE,
    ):
        self.errors_ = []
        self.val_errors_ = []
        best_state = (float("inf"), self.weights.copy(), self.bias)
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            errors = 0
            for features, label in training_data.items():
                inputs = np.array(features)
                prediction = self.predict(inputs)

                # Wrong prediction —> update weights
                if prediction != label:
                    update = self.learning_rate * (label - prediction)
                    self.weights += update * inputs
                    self.bias += update
                    errors += 1

            self.errors_.append(errors)

            best_state, epochs_without_improvement, stop = _early_stop_on_validation(
                self,
                validation_data,
                best_state,
                patience,
                epochs_without_improvement,
                epoch,
            )
            if stop:
                break

        # Restore weights from the best validation epoch
        if validation_data is not None:
            _, best_weights, best_bias = best_state
            self.weights = best_weights
            self.bias = best_bias


# Adaline learning rule
class Adaline:
    def __init__(self, learning_rate, epochs, n_features=NUMBER_OF_IRIS_PARAMETERS):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.uniform(0, 1, n_features)
        self.bias = random.uniform(0, 1)
        self.cost_ = []
        self.val_errors_ = []

    # Net input (before activation)
    def net_input(self, dataset_inputs):
        return np.dot(dataset_inputs, self.weights) + self.bias

    # Heaviside step activation
    def activation_function(self, vector_sum_of_neuron):
        if vector_sum_of_neuron >= 0.0:
            return 1.0
        else:
            return -1.0

    # Prediction
    def predict(self, dataset_inputs):
        return self.activation_function(self.net_input(dataset_inputs))

    # Training with early stopping on the validation set
    def train(
        self,
        training_data,
        validation_data=None,
        patience=EARLY_STOPPING_PATIENCE,
    ):
        self.cost_ = []
        self.val_errors_ = []
        best_state = (float("inf"), self.weights.copy(), self.bias)
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            epoch_cost = 0.0
            for features, label in training_data.items():
                inputs = np.array(features)

                net_output = self.net_input(inputs)
                error = label - net_output
                epoch_cost += error ** 2

                update = self.learning_rate * error
                self.weights += update * inputs
                self.bias += update

            self.cost_.append(0.5 * epoch_cost)

            best_state, epochs_without_improvement, stop = _early_stop_on_validation(
                self,
                validation_data,
                best_state,
                patience,
                epochs_without_improvement,
                epoch,
            )
            if stop:
                break

        if validation_data is not None:
            _, best_weights, best_bias = best_state
            self.weights = best_weights
            self.bias = best_bias


# 3-class classification (One-vs-Rest)
class OneVsRestClassifier:
    def __init__(self, classifier_class, classes, learning_rate, epochs, n_features=NUMBER_OF_IRIS_PARAMETERS):
        self.classes = classes
        self.classifiers = {}
        for c in classes:
            self.classifiers[c] = classifier_class(
                learning_rate, epochs, n_features=n_features
            )

    def train(self, training_data, validation_data=None, patience=EARLY_STOPPING_PATIENCE):
        for c in self.classes:
            binary_train_data = {
                features: (1.0 if label == c else -1.0)
                for features, label in training_data.items()
            }
            binary_val_data = None
            if validation_data is not None:
                binary_val_data = {
                    features: (1.0 if label == c else -1.0)
                    for features, label in validation_data.items()
                }
            self.classifiers[c].train(
                binary_train_data,
                validation_data=binary_val_data,
                patience=patience,
            )

    def predict(self, dataset_inputs):
        best_class = None
        max_activation = -float("inf")

        for c, clf in self.classifiers.items():
            activation = np.dot(dataset_inputs, clf.weights) + clf.bias
            if activation > max_activation:
                max_activation = activation
                best_class = c

        return best_class


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Model quality (accuracy)
def evaluate_model(model, dataset):
    correct = 0
    for features, true_label in dataset.items():
        if model.predict(np.array(features)) == true_label:
            correct += 1
    accuracy = (correct / len(dataset)) * 100 if len(dataset) > 0 else 0
    return correct, len(dataset), accuracy


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_decision_regions(
    X,
    y,
    classifier,
    resolution=0.02,
    xlabel=VIZ_FEATURE_LABELS[0],
    ylabel=VIZ_FEATURE_LABELS[1],
    title="",
    class_names=None,
    save_path=None,
):
    """
    Decision-region plot
    """
    y_plot = np.asarray(y)
    if y_plot.dtype.kind in {"U", "S", "O"}:
        unique_labels = list(dict.fromkeys(y_plot.tolist()))
    else:
        unique_labels = list(np.unique(y_plot))
    label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
    y_int = np.array([label_to_int[lab] for lab in y_plot])

    markers = ("s", "x", "o", "^", "v")
    colors = ("#FF6B6B", "#4D96FF", "#6BCB77", "#C4C4C4", "#9B59B6")
    cmap = ListedColormap(colors[: len(unique_labels)])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )
    grid = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = np.array([label_to_int[classifier.predict(p)] for p in grid]).reshape(xx1.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx1, xx2, Z, alpha=0.35, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    for idx, lab in enumerate(unique_labels):
        mask = y_int == idx
        display_name = (
            class_names[lab]
            if class_names and lab in class_names
            else str(lab)
        )
        scatter_kwargs = {
            "c": colors[idx],
            "marker": markers[idx % len(markers)],
            "label": display_name,
            "s": 55,
        }
        if markers[idx % len(markers)] != "x":
            scatter_kwargs["edgecolor"] = "black"
        ax.scatter(X[mask, 0], X[mask, 1], **scatter_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")
    plt.close(fig)


def plot_perceptron_errors(errors, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(errors) + 1), errors, marker="o")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Number of misclassifications")
    ax.set_title("Perceptron - Learning curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")
    plt.close(fig)


def plot_adaline_cost(cost, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(cost) + 1), cost, marker="o")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Sum-squared-error")
    ax.set_title("Adaline - Learning curve (SSE)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")
    plt.close(fig)


def plot_iris_scatter_3class(X, y, save_path=None):
    """Scatter of all three Iris species — highlights versicolor/virginica overlap."""
    markers = {"Iris-setosa": "s", "Iris-versicolor": "x", "Iris-virginica": "o"}
    colors = {
        "Iris-setosa": "#FF6B6B",
        "Iris-versicolor": "#4D96FF",
        "Iris-virginica": "#6BCB77",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for species in ("Iris-setosa", "Iris-versicolor", "Iris-virginica"):
        mask = y == species
        scatter_kwargs = {
            "c": colors[species],
            "marker": markers[species],
            "label": species,
            "s": 55,
        }
        if markers[species] != "x":
            scatter_kwargs["edgecolor"] = "black"
        ax.scatter(X[mask, 0], X[mask, 1], **scatter_kwargs)

    ax.set_xlabel(VIZ_FEATURE_LABELS[0])
    ax.set_ylabel(VIZ_FEATURE_LABELS[1])
    ax.set_title("Iris feature space (3 classes) — versicolor ≈ virginica")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")
    plt.close(fig)


def generate_plots(
    plots_dir,
    train_data_2d,
    val_data_2d,
    train_data_3_2d,
    val_data_3_2d,
    classes_list,
):
    os.makedirs(plots_dir, exist_ok=True)

    binary_names = {-1.0: "Iris-setosa", 1.0: "Iris-versicolor"}

    # --- Perceptron 2D ---
    ppn = Perceptron(
        learning_rate=LEARNING_FACTOR, epochs=NUMBER_OF_EPOCHS, n_features=2
    )
    ppn.train(train_data_2d, validation_data=val_data_2d)
    X_bin, y_bin = dataset_to_xy(train_data_2d)
    plot_decision_regions(
        X_bin,
        y_bin,
        ppn,
        title="Perceptron — decision regions (setosa vs versicolor)",
        class_names=binary_names,
        save_path=os.path.join(plots_dir, "perceptron_decision_regions.png"),
    )
    plot_perceptron_errors(
        ppn.errors_,
        save_path=os.path.join(plots_dir, "perceptron_learning_curve.png"),
    )

    # --- Adaline 2D ---
    ada = Adaline(
        learning_rate=LEARNING_FACTOR, epochs=NUMBER_OF_EPOCHS, n_features=2
    )
    ada.train(train_data_2d, validation_data=val_data_2d)
    plot_decision_regions(
        X_bin,
        y_bin,
        ada,
        title="Adaline — decision regions (setosa vs versicolor)",
        class_names=binary_names,
        save_path=os.path.join(plots_dir, "adaline_decision_regions.png"),
    )
    plot_adaline_cost(
        ada.cost_,
        save_path=os.path.join(plots_dir, "adaline_learning_curve.png"),
    )

    # --- 3-class scatter + One-vs-Rest decision regions ---
    X3, y3 = dataset_to_xy(train_data_3_2d)
    plot_iris_scatter_3class(
        X3,
        y3,
        save_path=os.path.join(plots_dir, "iris_3class_scatter.png"),
    )

    ovr = OneVsRestClassifier(
        classifier_class=Adaline,
        classes=classes_list,
        learning_rate=LEARNING_FACTOR,
        epochs=NUMBER_OF_EPOCHS,
        n_features=2,
    )
    ovr.train(train_data_3_2d, validation_data=val_data_3_2d)
    plot_decision_regions(
        X3,
        y3,
        ovr,
        title="One-vs-Rest Adaline — 3-class decision regions",
        save_path=os.path.join(plots_dir, "ovr_3class_decision_regions.png"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    filename = "iris_data.txt"

    num_setosa, num_versicolor = count_classes(filename)
    train_data, val_data, test_data = read_iris_data(
        filename, num_setosa, num_versicolor
    )

    # ----------------- PERCEPTRON TEST -----------------
    print("--- Perceptron learning ---")
    perceptron = Perceptron(learning_rate=LEARNING_FACTOR, epochs=NUMBER_OF_EPOCHS)
    perceptron.train(train_data, validation_data=val_data)

    c_val, t_val, acc_val = evaluate_model(perceptron, val_data)
    c_test, t_test, acc_test = evaluate_model(perceptron, test_data)
    print(f"Perceptron - Validation accuracy: {acc_val}% ({c_val}/{t_val})")
    print(f"Perceptron - Test accuracy: {acc_test}% ({c_test}/{t_test})\n")

    # ----------------- ADALINE TEST -----------------
    print("--- Adaline learning ---")
    adaline = Adaline(learning_rate=LEARNING_FACTOR, epochs=NUMBER_OF_EPOCHS)
    adaline.train(train_data, validation_data=val_data)

    c_val, t_val, acc_val = evaluate_model(adaline, val_data)
    c_test, t_test, acc_test = evaluate_model(adaline, test_data)
    print(f"Adaline - Validation accuracy: {acc_val}% ({c_val}/{t_val})")
    print(f"Adaline - Test accuracy: {acc_test}% ({c_test}/{t_test})\n")

    # ----------------- 3-CLASS TEST (One-vs-Rest) -----------------
    print("--- Training One-vs-Rest (3 classes) with Adaline ---")
    classes_list = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    train_data_3, val_data_3, test_data_3 = read_iris_data_3_classes(filename)

    ovr_classifier = OneVsRestClassifier(
        classifier_class=Adaline,
        classes=classes_list,
        learning_rate=LEARNING_FACTOR,
        epochs=NUMBER_OF_EPOCHS,
    )
    ovr_classifier.train(train_data_3, validation_data=val_data_3)

    c_val, t_val, acc_val = evaluate_model(ovr_classifier, val_data_3)
    c_test, t_test, acc_test = evaluate_model(ovr_classifier, test_data_3)
    print(f"One-vs-Rest - Validation accuracy: {acc_val}% ({c_val}/{t_val})")
    print(f"One-vs-Rest - Test accuracy: {acc_test}% ({c_test}/{t_test})")

    # ----------------- PLOTS  -----------------
    print("\n--- Generating plots ---")
    train_data_2d = subset_features(train_data)
    val_data_2d = subset_features(val_data)
    train_data_3_2d = subset_features(train_data_3)
    val_data_3_2d = subset_features(val_data_3)
    generate_plots(
        "plots",
        train_data_2d,
        val_data_2d,
        train_data_3_2d,
        val_data_3_2d,
        classes_list,
    )
