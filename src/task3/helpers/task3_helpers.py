from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK3_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TASK3_DIR / "data"
STOCKS_CSV = DATA_DIR / "all_stocks_5yr.csv"
PLOTS_DIR = TASK3_DIR / "plots"

# Data source
STOCKS_URL = (
    "https://raw.githubusercontent.com/stevejgordon/"
    "elasticsearch-examples/main/all_stocks_5yr.csv"
)

# Numeric indicators
FEATURE_COLUMNS = ("open", "high", "low", "close", "volume")

# Hyperparameter grid
PARAM_GRID = {
    "mlp__hidden_layer_sizes": [(n,) for n in range(20, 101, 20)],
    "mlp__activation": ["tanh", "logistic", "relu"],
    "mlp__learning_rate_init": [0.1, 0.01, 0.001],
    "mlp__learning_rate": ["constant", "adaptive"],
    "mlp__solver": ["adam", "lbfgs", "sgd"],
}

DEFAULT_TICKER = "AAPL"
DEFAULT_K = 4  # → 5 trading days of history
DEFAULT_L = 4  # close price among FEATURE_COLUMNS
DEFAULT_TEST_PCT = 20  # K% last samples = test set
DEFAULT_VAL_PCT = 15  # validation block just before test (for GridSearch)
MAX_ITER = 500

# Different values of K
TEST_PCT_CANDIDATES = (10, 15, 20, 25, 30, 40)
