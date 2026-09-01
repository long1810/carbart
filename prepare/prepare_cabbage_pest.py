import os
import shutil
import random
from pathlib import Path

# =========================
# CẤU HÌNH
# =========================
SOURCE_DIR = Path("dataset_pest")
OUTPUT_DIR = Path("dataset_pest_split")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def is_image_file(file_path):
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def create_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def split_dataset():
    random.seed(SEED)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {SOURCE_DIR}")

    classes = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]

    if len(classes) == 0:
        raise ValueError("Không tìm thấy class nào trong dataset_pest/")

    print("Các class tìm thấy:")
    for cls in classes:
        print("-", cls.name)

    for class_dir in classes:
        class_name = class_dir.name

        images = [f for f in class_dir.iterdir() if f.is_file() and is_image_file(f)]
        random.shuffle(images)

        total = len(images)

        if total == 0:
            print(f"Bỏ qua class {class_name} vì không có ảnh.")
            continue

        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        train_files = images[:train_count]
        val_files = images[train_count:train_count + val_count]
        test_files = images[train_count + val_count:]

        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        for split_name, files in splits.items():
            target_class_dir = OUTPUT_DIR / split_name / class_name
            create_dir(target_class_dir)

            for file_path in files:
                target_path = target_class_dir / file_path.name
                shutil.copy2(file_path, target_path)

        print(
            f"{class_name}: "
            f"train={len(train_files)}, "
            f"val={len(val_files)}, "
            f"test={len(test_files)}"
        )

    print("\nChia dataset hoàn tất.")
    print(f"Dataset mới nằm tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    split_dataset()