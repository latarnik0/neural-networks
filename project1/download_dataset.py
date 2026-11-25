from __future__ import annotations

import shutil
import sys
from pathlib import Path

DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
CLASSES = ("glioma", "meningioma", "pituitary", "notumor")

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "brain_tumor_mri"


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

    for item in cache_path.iterdir():
        dest = DATA_DIR / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    return DATA_DIR


def summarize(root: Path) -> None:
    print(f"Dataset root: {root}")
    for split in ("Training", "Testing"):
        split_dir = root / split
        if not split_dir.is_dir():
            alt = root / split.lower()
            split_dir = alt if alt.is_dir() else split_dir
        if not split_dir.is_dir():
            print(f"  missing split: {split}")
            continue
        print(f"  {split_dir.name}/")
        for cls in CLASSES:
            cls_dir = split_dir / cls
            n = len(list(cls_dir.glob("*"))) if cls_dir.is_dir() else 0
            print(f"    {cls}: {n} images")


def main() -> None:
    path = download()
    summarize(path)


if __name__ == "__main__":
    main()
