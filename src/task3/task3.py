"""
MLP regression / approximation on S&P 500 stock indicators.

Builds a supervised set from a multivariate time series (open, high, low,
close, volume). With lookback parameter k and target
indicator index l (1-based).
Train = first (100-K)% of samples chronologically; test = last K%.
Inputs are scaled per feature to [0, 1] (MinMaxScaler).
Model: MLPRegressor, one hidden layer, linear output.
Hyperparameters via GridSearchCV.
"""

from __future__ import annotations

import csv
import urllib.request
import warnings

from helpers.task3_helpers import *

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def ensure_stocks_csv(path=STOCKS_CSV, url=STOCKS_URL):
    """Download all_stocks_5yr.csv."""
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading S&P 500 data → {path}")
    urllib.request.urlretrieve(url, path)
    print("Download done")
    return path


def load_ticker_series(csv_path, ticker, feature_columns=FEATURE_COLUMNS):
    """Load one ticker as (N, r) float array, chronological order."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Name"] != ticker:
                continue
            values = []
            skip = False
            for col in feature_columns:
                raw = row[col]
                if raw is None or raw == "":
                    skip = True
                    break
                values.append(float(raw))
            if not skip:
                rows.append(values)
    if not rows:
        raise ValueError(f"No rows for ticker {ticker!r} in {csv_path}")
    return np.asarray(rows, dtype=float)


def build_regression_dataset(series, k, l):
    n_steps, r = series.shape
    if not (1 <= l <= r):
        raise ValueError(f"l must be in 1..{r}, got {l}")
    if k < 0:
        raise ValueError("k must be ≥ 0")

    n_samples = n_steps - (k + 1)
    if n_samples <= 0:
        raise ValueError(
            f"Series too short: N={n_steps}, need at least k+2={k + 2}"
        )

    window = k + 1
    X = np.empty((n_samples, window * r), dtype=float)
    y = np.empty(n_samples, dtype=float)
    target_col = l - 1

    for i in range(n_samples):
        X[i] = series[i : i + window].ravel()
        y[i] = series[i + window, target_col]
    return X, y


def chronological_split(X, y, test_pct):
    """First (100-K)% train, last K% test."""
    if not (0 < test_pct < 100):
        raise ValueError(f"test_pct must be in (0, 100), got {test_pct}")
    n = len(X)
    n_test = max(1, int(round(n * test_pct / 100.0)))
    n_train = n - n_test
    if n_train < 1:
        raise ValueError(f"test_pct={test_pct} leaves empty train set (n={n})")
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_mlp_pipeline(max_iter=MAX_ITER):
    """MinMaxScaler [0, 1] per feature + MLPRegressor"""
    return Pipeline(
        [
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            (
                "mlp",
                MLPRegressor(
                    max_iter=max_iter,
                    validation_fraction = 0.2,
                    random_state=42,
                    shuffle=True,
                    early_stopping=False,
                ),
            ),
        ]
    )


def run_grid_search(X_train, y_train, max_iter=MAX_ITER, cv=3, n_jobs=-1):
    grid = GridSearchCV(
        estimator=make_mlp_pipeline(max_iter),
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "r2": r2_score(y_true, y_pred),
    }


def print_best_params(grid, label):
    print(f"\n=== {label}: best hyperparameters ===")
    for key, value in grid.best_params_.items():
        print(f"  {key}: {value}")
    best_mse = -grid.best_score_
    print(f"  Best MSE: {best_mse:.6f}")
    print(f"  Best RMSE: {np.sqrt(best_mse):.6f}")
    mlp = grid.best_estimator_.named_steps["mlp"]
    print(f"  Output activation: {mlp.out_activation_}")


def evaluate(model, X, y, split_name):
    y_pred = model.predict(X)
    m = regression_metrics(y, y_pred)
    print(
        f"  {split_name}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, "
        f"R²={m['r2']:.4f}"
    )
    return y_pred, m


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_predictions(y_true, y_pred, title, save_path, target_name="close"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    idx = np.arange(len(y_true))
    axes[0].plot(idx, y_true, label="true", linewidth=1.5)
    axes[0].plot(idx, y_pred, label="pred", linewidth=1.2, alpha=0.85)
    axes[0].set_ylabel(f"{target_name} [$]")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_true, y_pred, s=12, alpha=0.6, edgecolors="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    axes[1].plot([lo, hi], [lo, hi], "k--", linewidth=1, label="ideal")
    axes[1].set_xlabel(f"true {target_name}")
    axes[1].set_ylabel(f"predicted {target_name}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal", adjustable="box")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_residuals(y_true, y_pred, save_path):
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.plot(residuals, linewidth=1)
    ax.set_xlabel("test sample index")
    ax.set_ylabel("pred − true")
    ax.set_title("Test residuals")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_k_sweep(results, save_path):
    """K% test horizon vs test metrics — pick workable forward window."""
    ks = [r["test_pct"] for r in results]
    maes = [r["mae"] for r in results]
    r2s = [r["r2"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_mae = "#FF6B6B"
    color_r2 = "#4D96FF"

    ax1.plot(ks, maes, "o-", color=color_mae, linewidth=2, label="MAE")
    ax1.set_xlabel("K — test set [% of samples, last K%]")
    ax1.set_ylabel("Test MAE", color=color_mae)
    ax1.tick_params(axis="y", labelcolor=color_mae)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ks, r2s, "s-", color=color_r2, linewidth=2, label="R²")
    ax2.set_ylabel("Test R²", color=color_r2)
    ax2.tick_params(axis="y", labelcolor=color_r2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title("Forward-test horizon K: error vs predictability")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_loss_curve(pipeline, save_path):
    mlp = pipeline.named_steps["mlp"]
    loss = getattr(mlp, "loss_curve_", None)
    if not loss:
        print("  No loss_curve_ (solver may be lbfgs) — skip loss plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(loss) + 1), loss, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss")
    ax.set_title("MLPRegressor learning curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# K selection
# ---------------------------------------------------------------------------

def choose_k(results, r2_floor=0.0):
    """
    Largest K whose test R² stays ≥ max(r2_floor, 0.9 · best_r2).
    Prefer farther forecast only while quality holds.
    """
    if not results:
        raise ValueError("empty K sweep")
    best_r2 = max(r["r2"] for r in results)
    threshold = max(r2_floor, 0.9 * best_r2)
    admissible = [r for r in results if r["r2"] >= threshold]
    chosen = max(admissible, key=lambda r: r["test_pct"])
    return chosen, threshold


def sweep_test_pct(
    X,
    y,
    best_params,
    test_pcts=TEST_PCT_CANDIDATES,
    max_iter=MAX_ITER,
):
    """Refit best MLP for each chronological K% split; collect test metrics."""
    mlp_params = {
        k.replace("mlp__", ""): v for k, v in best_params.items()
    }
    results = []
    for pct in test_pcts:
        X_tr, X_te, y_tr, y_te = chronological_split(X, y, pct)
        pipe = make_mlp_pipeline(max_iter)
        pipe.set_params(**{f"mlp__{k}": v for k, v in mlp_params.items()})
        pipe.fit(X_tr, y_tr)
        y_hat = pipe.predict(X_te)
        m = regression_metrics(y_te, y_hat)
        results.append(
            {
                "test_pct": pct,
                "n_train": len(X_tr),
                "n_test": len(X_te),
                "pipeline": pipe,
                "y_true": y_te,
                "y_pred": y_hat,
                **m,
            }
        )
        print(
            f"  K={pct}% (train={len(X_tr)}, test={len(X_te)}): "
            f"MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, R²={m['r2']:.4f}"
        )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    ticker = DEFAULT_TICKER
    k = DEFAULT_K
    l = DEFAULT_L
    target_name = FEATURE_COLUMNS[l - 1]

    print("--- MLPRegressor on S&P 500 indicators ---")
    csv_path = ensure_stocks_csv()
    series = load_ticker_series(csv_path, ticker)
    print(
        f"Ticker={ticker}, N={len(series)}, r={series.shape[1]} "
        f"features={FEATURE_COLUMNS}"
    )
    print(
        f"Window: days i..i+{k} -> predict day i+{k + 1} "
        f"indicator l={l} ({target_name})"
    )

    X, y = build_regression_dataset(series, k=k, l=l)
    print(f"Supervised samples: {len(X)}, input dim={X.shape[1]}")

    # --- Grid search at reference K ---
    ref_pct = DEFAULT_TEST_PCT
    X_train, X_test, y_train, y_test = chronological_split(X, y, ref_pct)
    print(
        f"\nGridSearchCV at K={ref_pct}%: "
        f"train={len(X_train)}, test={len(X_test)}"
    )
    print(
        f"Param grid size: "
        f"{np.prod([len(v) for v in PARAM_GRID.values()])} configs × 3-fold CV"
    )

    grid = run_grid_search(X_train, y_train)
    print_best_params(grid, f"{ticker} @ K={ref_pct}%")

    best = grid.best_estimator_
    print("\nReference split evaluation:")
    evaluate(best, X_train, y_train, "train")
    y_pred_ref, test_m = evaluate(best, X_test, y_test, "test")

    plot_predictions(
        y_test,
        y_pred_ref,
        title=(
            f"{ticker} — test predictions (K={ref_pct}%, "
            f"R²={test_m['r2']:.3f})"
        ),
        save_path=PLOTS_DIR / "predictions_vs_true.png",
        target_name=target_name,
    )
    plot_residuals(y_test, y_pred_ref, PLOTS_DIR / "test_residuals.png")
    plot_loss_curve(best, PLOTS_DIR / "learning_curve.png")

    # --- Sweep K to find how far ahead we can still predict ---
    print("\n--- Sweep K (last K% as forward test) ---")
    sweep = sweep_test_pct(X, y, grid.best_params_)
    plot_k_sweep(sweep, PLOTS_DIR / "k_horizon_metrics.png")

    chosen, threshold = choose_k(sweep)
    print(
        f"\nChosen K={chosen['test_pct']}% "
        f"(R²={chosen['r2']:.4f}, MAE={chosen['mae']:.4f}; "
        f"threshold R²≥{threshold:.4f})"
    )
    print(
        "Interpretation: last "
        f"{chosen['test_pct']}% of the series remains predictable "
        "at near-peak quality under the selected MLP."
    )

    plot_predictions(
        chosen["y_true"],
        chosen["y_pred"],
        title=(
            f"{ticker} — chosen K={chosen['test_pct']}% "
            f"(R²={chosen['r2']:.3f})"
        ),
        save_path=PLOTS_DIR / "predictions_chosen_k.png",
        target_name=target_name,
    )
    print("\nDone.")
