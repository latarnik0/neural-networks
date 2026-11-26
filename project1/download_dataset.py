from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

DATASET = "nikhil7280/weather-type-classification"
CSV_NAME = "weather_classification_data.csv"

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "weather_type"


def _ensure_kagglehub() -> None:
    try:
        import kagglehub  # noqa: F401
    except ImportError:
        print("Installing kagglehub...", file=sys.stderr)
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "kagglehub"],
        )


def download() -> Path:
    _ensure_kagglehub()
    import kagglehub

    cache_path = Path(kagglehub.dataset_download(DATASET))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    source = cache_path / CSV_NAME
    if not source.is_file():
        matches = list(cache_path.rglob(CSV_NAME))
        if not matches:
            raise FileNotFoundError(f"{CSV_NAME} not found in {cache_path}")
        source = matches[0]

    dest = DATA_DIR / CSV_NAME
    shutil.copy2(source, dest)
    return dest


def summarize(csv_path: Path) -> None:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    weather_type_counts: dict[str, int] = {}
    for row in rows:
        weather_type = row["Weather Type"]
        weather_type_counts[weather_type] = weather_type_counts.get(weather_type, 0) + 1

    print(f"Dataset file: {csv_path}")
    print(f"  rows: {len(rows)}")
    print(f"  columns: {', '.join(fieldnames)}")
    print("  Weather Type:")
    for weather_type in sorted(weather_type_counts):
        print(f"    {weather_type}: {weather_type_counts[weather_type]}")


def main() -> None:
    path = download()
    summarize(path)


if __name__ == "__main__":
    main()
