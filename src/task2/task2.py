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
from matplotlib.colors import ListedColormap
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 2D Iris view for decision-region plot (sepal length, petal length)
IRIS_VIZ_FEATURES = (0, 2)
IRIS_VIZ_LABELS = ("sepal length [cm]", "petal length [cm]")


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
    """
    Load MNIST (28×28 → 784 features) via OpenML.
    Official 10k test; stratified validation carved from the 60k train set.
    """
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


def plot_iris_decision_regions(
    pipeline,
    X,
    y,
    class_labels,
    save_path,
    feature_names=IRIS_VIZ_LABELS,
    resolution=0.02,
):
    """
    2D decision regions for the 3 Iris species (trained MLP).
    With a hidden layer the boundary is generally nonlinear — not a single
    hyperplane — but this is the standard visualization of how the classes
    are separated in feature space.
    """
    markers = ("s", "x", "o")
    colors = ("#FF6B6B", "#4D96FF", "#6BCB77")
    cmap = ListedColormap(colors[: len(class_labels)])

    label_to_int = {lab: i for i, lab in enumerate(class_labels)}
    y_int = np.array([label_to_int[lab] for lab in y])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    Z = pipeline.predict(grid)
    Z = np.array([label_to_int[lab] for lab in Z]).reshape(xx1.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx1, xx2, Z, alpha=0.35, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    for idx, lab in enumerate(class_labels):
        mask = y_int == idx
        scatter_kwargs = {
            "c": colors[idx],
            "marker": markers[idx],
            "label": lab,
            "s": 55,
        }
        if markers[idx] != "x":
            scatter_kwargs["edgecolor"] = "black"
        ax.scatter(X[mask, 0], X[mask, 1], **scatter_kwargs)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("Iris — MLP decision regions (3 classes)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_iris_mlp_decision_boundary(best_params, X_train, y_train, X_val, y_val, X_test, y_test, class_labels):
    """
    Retrain MLP on 2 features (sepal/petal length) with the best grid
    hyperparameters and plot the decision regions separating the 3 species.
    """
    feat = list(IRIS_VIZ_FEATURES)
    X_tr = np.vstack([X_train[:, feat], X_val[:, feat], X_test[:, feat]])
    y_tr = np.concatenate([y_train, y_val, y_test])

    # Fit on train+val (same features used for the mesh); points include test
    X_fit = np.vstack([X_train[:, feat], X_val[:, feat]])
    y_fit = np.concatenate([y_train, y_val])

    pipe = make_mlp_pipeline(max_iter=IRIS_MAX_ITER)
    pipe.set_params(**{f"mlp__{k}": v for k, v in best_params.items()})
    pipe.fit(X_fit, y_fit)

    plot_iris_decision_regions(
        pipe,
        X_tr,
        y_tr,
        class_labels,
        save_path=PLOTS_DIR / "iris_decision_regions.png",
    )


# ---------------------------------------------------------------------------
# MNIST hyperparameter comparison plots
# ---------------------------------------------------------------------------

def _fit_mnist_variant(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    *,
    hidden_layer_sizes,
    activation,
    solver,
    learning_rate_init=0.01,
    max_iter=MNIST_MAX_ITER,
    force_full_iters=False,
):
    """Train one MLP configuration; return metrics, loss curve and wall time."""
    import time

    pipe = make_mlp_pipeline(max_iter=max_iter)
    pipe.set_params(
        mlp__hidden_layer_sizes=hidden_layer_sizes,
        mlp__activation=activation,
        mlp__solver=solver,
        mlp__learning_rate_init=learning_rate_init,
    )
    # Fair timing: do not stop early — compare equal iteration budgets
    if force_full_iters:
        pipe.set_params(
            mlp__tol=1e-20,
            mlp__n_iter_no_change=max_iter + 1,
        )

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    mlp = pipe.named_steps["mlp"]
    loss_curve = list(getattr(mlp, "loss_curve_", []) or [])
    n_iter = getattr(mlp, "n_iter_", len(loss_curve))

    return {
        "pipeline": pipe,
        "elapsed_s": elapsed,
        "n_iter": n_iter,
        "loss_curve": loss_curve,
        "train_acc": accuracy_score(y_train, pipe.predict(X_train)),
        "val_acc": accuracy_score(y_val, pipe.predict(X_val)),
        "test_acc": accuracy_score(y_test, pipe.predict(X_test)),
        "hidden": hidden_layer_sizes[0],
        "activation": activation,
        "solver": solver,
    }


def plot_hidden_size_comparison(results, save_path):
    """Why few neurons underfit: accuracy vs hidden layer size."""
    sizes = [r["hidden"] for r in results]
    val_accs = [r["val_acc"] * 100 for r in results]
    test_accs = [r["test_acc"] * 100 for r in results]

    x = np.arange(len(sizes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, val_accs, width, label="Validation")
    ax.bar(x + width / 2, test_accs, width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Hidden neurons")
    ax.set_ylabel("Accuracy [%]")
    ax.set_title("MNIST — capacity: more neurons → higher accuracy")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_confusion_20_vs_100(res_20, res_100, X_test, y_test, save_path):
    """Side-by-side confusion matrices: underpowered (20) vs stronger (100)."""
    labels = list(range(10))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, res, title in [
        (axes[0], res_20, "20 hidden neurons"),
        (axes[1], res_100, "100 hidden neurons"),
    ]:
        cm = confusion_matrix(
            y_test, res["pipeline"].predict(X_test), labels=labels
        )
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(
            f"{title}\ntest acc = {res['test_acc'] * 100:.1f}%"
        )

    fig.suptitle("MNIST — fewer neurons → more off-diagonal mistakes")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_activation_learning_curves(relu_res, tanh_res, save_path):
    """ReLU typically descends faster than tanh on the same setup."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        range(1, len(relu_res["loss_curve"]) + 1),
        relu_res["loss_curve"],
        marker="o",
        markersize=3,
        label=(
            f"ReLU ({relu_res['n_iter']} iters, "
            f"{relu_res['elapsed_s']:.1f}s, "
            f"test={relu_res['test_acc'] * 100:.1f}%)"
        ),
    )
    ax.plot(
        range(1, len(tanh_res["loss_curve"]) + 1),
        tanh_res["loss_curve"],
        marker="s",
        markersize=3,
        label=(
            f"tanh ({tanh_res['n_iter']} iters, "
            f"{tanh_res['elapsed_s']:.1f}s, "
            f"test={tanh_res['test_acc'] * 100:.1f}%)"
        ),
    )

    # Mark first time each curve reaches a modest loss threshold
    threshold = 0.2
    for res, name, style in (
        (relu_res, "ReLU", "--"),
        (tanh_res, "tanh", ":"),
    ):
        hit = next(
            (i + 1 for i, v in enumerate(res["loss_curve"]) if v <= threshold),
            None,
        )
        if hit is not None:
            ax.axvline(
                hit, color="gray", linestyle=style, alpha=0.7,
                label=f"{name} reaches loss≤{threshold} at iter {hit}",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss")
    ax.set_title("MNIST — ReLU learns faster than tanh (loss curves)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_solver_runtime(fixed_width_results, scaling_results, save_path):
    """
    Solver wall-clock cost (equal iteration budget):
      - left: adam / sgd / lbfgs at width 100
      - right: time vs hidden size for adam vs lbfgs
    """
    names = [r["solver"] for r in fixed_width_results]
    times = [r["elapsed_s"] for r in fixed_width_results]
    colors = {"adam": "#4D96FF", "sgd": "#6BCB77", "lbfgs": "#FF6B6B"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    bar_colors = [colors.get(n, "#888888") for n in names]
    axes[0].bar(names, times, color=bar_colors)
    axes[0].set_ylabel("Wall-clock time [s]")
    axes[0].set_title("Training time @ 100 hidden (same #iters)")
    axes[0].grid(True, axis="y", alpha=0.3)
    for i, (t, r) in enumerate(zip(times, fixed_width_results)):
        axes[0].text(
            i, t, f"{t:.1f}s\n({r['n_iter']} it)",
            ha="center", va="bottom", fontsize=8,
        )

    for solver in ("adam", "lbfgs"):
        xs = [r["hidden"] for r in scaling_results if r["solver"] == solver]
        ys = [r["elapsed_s"] for r in scaling_results if r["solver"] == solver]
        axes[1].plot(
            xs, ys, marker="o", linewidth=2, label=solver, color=colors[solver]
        )
    axes[1].set_xlabel("Hidden neurons")
    axes[1].set_ylabel("Wall-clock time [s]")
    axes[1].set_title("Time vs width (equal iteration budget)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("MNIST — solver computational cost (sklearn MLPClassifier)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def run_mnist_hyperparameter_visualizations(
    X_train, y_train, X_val, y_val, X_test, y_test, plots_dir=PLOTS_DIR
):
    """
    Controlled one-factor comparisons on MNIST:
      1) hidden size 20…100 — capacity / underfitting
      2) ReLU vs tanh — learning speed (loss curves)
      3) adam vs lbfgs (+ sgd) — computational cost
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Use a stratified subset so experiments finish in reasonable time
    X_s, _, y_s, _ = train_test_split(
        X_train,
        y_train,
        train_size=min(MNIST_GRID_SAMPLES, len(X_train)),
        random_state=42,
        stratify=y_train,
    )
    print(f"\n--- MNIST hyperparameter visualizations (n_train={len(X_s)}) ---")

    # 1) Hidden layer size
    print("Comparing hidden layer sizes...")
    neuron_results = []
    for n in range(20, 101, 20):
        print(f"  hidden=({n},), relu, adam...")
        neuron_results.append(
            _fit_mnist_variant(
                X_s, y_s, X_val, y_val, X_test, y_test,
                hidden_layer_sizes=(n,),
                activation="relu",
                solver="adam",
            )
        )
        r = neuron_results[-1]
        print(
            f"    test={r['test_acc'] * 100:.2f}%, "
            f"val={r['val_acc'] * 100:.2f}%, time={r['elapsed_s']:.1f}s"
        )

    plot_hidden_size_comparison(
        neuron_results, plots_dir / "mnist_hidden_size_accuracy.png"
    )
    plot_confusion_20_vs_100(
        neuron_results[0],
        neuron_results[-1],
        X_test,
        y_test,
        plots_dir / "mnist_confusion_20_vs_100.png",
    )

    # 2) Activation: ReLU vs tanh (same width / solver / lr)
    print("Comparing activations (ReLU vs tanh)...")
    relu_res = _fit_mnist_variant(
        X_s, y_s, X_val, y_val, X_test, y_test,
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
    )
    tanh_res = _fit_mnist_variant(
        X_s, y_s, X_val, y_val, X_test, y_test,
        hidden_layer_sizes=(100,),
        activation="tanh",
        solver="adam",
    )
    print(
        f"  ReLU: loss@{relu_res['n_iter']}="
        f"{relu_res['loss_curve'][-1]:.4f}, time={relu_res['elapsed_s']:.1f}s"
    )
    print(
        f"  tanh: loss@{tanh_res['n_iter']}="
        f"{tanh_res['loss_curve'][-1]:.4f}, time={tanh_res['elapsed_s']:.1f}s"
    )
    plot_activation_learning_curves(
        relu_res, tanh_res, plots_dir / "mnist_relu_vs_tanh_learning.png"
    )

    # 3) Solver cost: adam vs lbfgs vs sgd (+ scaling with width)
    # Equal iteration budget so wall time reflects cost/iteration (lbfgs heavier)
    print("Comparing solvers (runtime, fixed iteration budget)...")
    X_sol, _, y_sol, _ = train_test_split(
        X_train,
        y_train,
        train_size=min(12000, len(X_train)),
        random_state=0,
        stratify=y_train,
    )
    solver_iters = 40
    print(f"  Solver timing on n_train={len(X_sol)}, max_iter={solver_iters}")

    solver_results = []
    for solver in ("adam", "sgd", "lbfgs"):
        print(f"  solver={solver}, hidden=100...")
        solver_results.append(
            _fit_mnist_variant(
                X_sol, y_sol, X_val, y_val, X_test, y_test,
                hidden_layer_sizes=(100,),
                activation="relu",
                solver=solver,
                max_iter=solver_iters,
                force_full_iters=True,
            )
        )
        r = solver_results[-1]
        print(
            f"    test={r['test_acc'] * 100:.2f}%, "
            f"time={r['elapsed_s']:.1f}s, iters={r['n_iter']}"
        )

    # Width scaling under the same iteration budget
    print("  Scaling adam vs lbfgs with hidden size (fixed iters)...")
    scaling_results = []
    for n in (20, 50, 100, 200):
        for solver in ("adam", "lbfgs"):
            print(f"    solver={solver}, hidden={n}...")
            scaling_results.append(
                _fit_mnist_variant(
                    X_sol, y_sol, X_val, y_val, X_test, y_test,
                    hidden_layer_sizes=(n,),
                    activation="relu",
                    solver=solver,
                    max_iter=solver_iters,
                    force_full_iters=True,
                )
            )
            r = scaling_results[-1]
            print(f"      time={r['elapsed_s']:.1f}s, iters={r['n_iter']}")

    plot_solver_runtime(
        solver_results,
        scaling_results,
        plots_dir / "mnist_solver_runtime.png",
    )
    print("Hyperparameter visualizations done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------- Part 1: Iris -----------------
    print("--- Part 1: Iris (MLP + GridSearchCV) ---")
    X_train, y_train, X_val, y_val, X_test, y_test = load_iris_xy()
    iris_labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    print(
        f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    iris_grid = run_grid_search(X_train, y_train, max_iter=IRIS_MAX_ITER, cv=3)
    print_best_params(iris_grid, "Iris")

    best_iris = iris_grid.best_estimator_
    val_acc = accuracy_score(y_val, best_iris.predict(X_val))
    print(f"  Validation accuracy: {val_acc * 100:.2f}%")

    evaluate_and_plot(
        best_iris,
        X_test,
        y_test,
        title="Iris — MLP confusion matrix",
        save_path=PLOTS_DIR / "iris_confusion_matrix.png",
        labels=iris_labels,
    )

    # Decision regions in 2D (sepal length × petal length)
    iris_best_params = {
        k.replace("mlp__", ""): v for k, v in iris_grid.best_params_.items()
    }
    print("Plotting Iris MLP decision regions (2 features)...")
    plot_iris_mlp_decision_boundary(
        iris_best_params,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        iris_labels,
    )

    # ----------------- Part 2: MNIST -----------------
    print("\n--- Part 2: MNIST (MLP + GridSearchCV) ---")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_xy()
    digit_labels = list(range(10))

    print(
        f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    # Grid search on a stratified subset (full 90×CV on ~51k is very slow)
    X_grid, _, y_grid, _ = train_test_split(
        X_train,
        y_train,
        train_size=min(MNIST_GRID_SAMPLES, len(X_train)),
        random_state=42,
        stratify=y_train,
    )
    print(f"Grid search on {len(X_grid)} stratified samples")

    mnist_grid = run_grid_search(X_grid, y_grid, max_iter=MNIST_MAX_ITER, cv=3)
    print_best_params(mnist_grid, "MNIST")

    # Refit best hyperparameters on the full training set
    best_params = {
        k.replace("mlp__", ""): v for k, v in mnist_grid.best_params_.items()
    }
    final_mnist = make_mlp_pipeline(max_iter=MNIST_MAX_ITER)
    final_mnist.set_params(**{f"mlp__{k}": v for k, v in best_params.items()})
    print("Refitting best MNIST model on full training set...")
    final_mnist.fit(X_train, y_train)

    val_acc = accuracy_score(y_val, final_mnist.predict(X_val))
    print(f"  Validation accuracy: {val_acc * 100:.2f}%")

    evaluate_and_plot(
        final_mnist,
        X_test,
        y_test,
        title="MNIST — MLP confusion matrix",
        save_path=PLOTS_DIR / "mnist_confusion_matrix.png",
        labels=digit_labels,
    )

    # Controlled comparisons: neurons / activation / solver
    run_mnist_hyperparameter_visualizations(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print("\nDone.")
