import csv
import random
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data/weather_type"
SOURCE_CSV_PATH = DATA_PATH / "weather_classification_data.csv"
TRAINING_SET_PATH = DATA_PATH / "Training" / "training_weather.csv"
VALIDATION_SET_PATH = DATA_PATH / "Validation" / "validation_weather.csv"
TESTING_SET_PATH = DATA_PATH / "Testing" / "testing_weather.csv"

TARGET_COLUMN = "Weather Type"
IGNORED_COLUMNS = {TARGET_COLUMN}
TARGET_LABELS = ("Cloudy", "Rainy", "Snowy", "Sunny")
CATEGORICAL_LABELS = {
    "Cloud Cover": ("clear", "cloudy", "overcast", "partly cloudy"),
    "Season": ("Autumn", "Spring", "Summer", "Winter"),
    "Location": ("coastal", "inland", "mountain"),
}

ONE_HOT_CLOUD_COVER = {"clear": 0, "cloudy": 1, "overcast": 2, "partly cloudy": 3}
ONE_HOT_SEASON = {"Autumn": 0, "Spring": 1, "Summer": 2, "Winter": 3}
ONE_HOT_LOCATION = {"coastal": 0, "inland": 1, "mountain": 2}

def clean_humidity_data():
    df = pd.read_csv('weather_classification_data.csv')
    df = df[df['Humidity'] < 101]
    df.to_csv('weather_classification_data_cleaned.csv', index=False)

def read_csv(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_by_class(rows, validation_set_ratio, test_set_ratio):
    rows_by_class = {}
    for row in rows:
        rows_by_class.setdefault(row[TARGET_COLUMN], []).append(row)

    training_rows = []
    validation_rows = []
    test_rows = []
    for class_rows in rows_by_class.values():
        random.shuffle(class_rows)
        validation_count = int(len(class_rows) * validation_set_ratio)
        test_count = int(len(class_rows) * test_set_ratio)
        validation_end = validation_count + test_count
        validation_rows.extend(class_rows[:validation_count])
        test_rows.extend(class_rows[validation_count:validation_end])
        training_rows.extend(class_rows[validation_end:])
    return training_rows, validation_rows, test_rows


def write_data_splits(validation_set_ratio, test_set_ratio, seed):
    random.seed(seed)
    fieldnames, rows = read_csv(SOURCE_CSV_PATH)
    training_rows, validation_rows, test_rows = split_by_class(
        rows, validation_set_ratio, test_set_ratio
    )
    write_csv(TRAINING_SET_PATH, fieldnames, training_rows)
    write_csv(VALIDATION_SET_PATH, fieldnames, validation_rows)
    write_csv(TESTING_SET_PATH, fieldnames, test_rows)


def create_validation_set(validation_set_ratio: float = 0.15, seed: int = 37):
    write_data_splits(validation_set_ratio, 0.15, seed)


def create_test_set(test_set_ratio: float = 0.15, seed: int = 37):
    write_data_splits(0.15, test_set_ratio, seed)


def feature_value(name, value):
    labels = CATEGORICAL_LABELS.get(name)
    if labels is None:
        return float(value)
    return float(labels.index(value))


def load_split(path: Path):
    fieldnames, rows = read_csv(path)
    feature_names = [name for name in fieldnames if name not in IGNORED_COLUMNS]
    features = np.array(
        [[feature_value(name, row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    targets = np.array(
        [TARGET_LABELS.index(row[TARGET_COLUMN]) for row in rows], dtype=np.int64
    )
    return features, targets


def data_preprocessing():
    x_train, y_train = load_split(TRAINING_SET_PATH)
    x_val, y_val = load_split(VALIDATION_SET_PATH)

    x_min = x_train.min(axis=0)
    x_max = x_train.max(axis=0)
    span = x_max - x_min
    span[span == 0] = 1.0

    x_train = (x_train - x_min) / span
    x_val = (x_val - x_min) / span
    return x_train, y_train, x_val, y_val


if __name__ == "__main__":
    clean_humidity_data()
    create_validation_set()
    create_test_set()
