import os
from pathlib import Path
import shutil
import random
from PIL import Image
import numpy as np

PROJECT_ROOT_PATH = Path("/home/latarnik3/studies/neural_networks/project1")
DATA_PATH = PROJECT_ROOT_PATH / "data/brain_tumor_mri"
TESTING_SET_PATH = DATA_PATH / "Testing"
TRAINING_SET_PATH = DATA_PATH / "Training"
VALIDATION_SET_PATH = DATA_PATH / "Validation"


def create_validation_set(validation_set_ratio: float = 0.15, seed: int = 37):
    data_classes = ["glioma", "meningioma", "notumor", "pituitary"]
    random.seed(seed)

    for data_class in data_classes:
        try:
            os.makedirs(VALIDATION_SET_PATH / data_class, exist_ok=True)
        except Exception as e:
            print(f'Cannot create {data_class} subdirectory!', e)

    for dir in TRAINING_SET_PATH.iterdir():
        if dir.is_dir():
            number_of_files = sum(1 for file in dir.iterdir() if file.is_file())
            temp = 1
            files = [f for f in dir.iterdir() if f.is_file()]
            random.shuffle(files)
            for file in dir.iterdir():
                if temp > number_of_files * validation_set_ratio:
                    break
                if file.is_file():
                    try:
                        shutil.move(file, VALIDATION_SET_PATH / dir.name / file.name)
                    except Exception as e:
                        print("Error occurred while copying: ", e)
                    temp += 1


def data_preprocessing():
    for path in DATA_PATH.rglob("*"):
        if not path.is_file():
            continue

        with Image.open(path) as img:
            img = img.convert("RGB").resize((256, 256))
            raw_image_matrix = np.array(img)
            processed_image_matrix = raw_image_matrix / 255.0

        

if __name__ == "__main__":
    create_validation_set()