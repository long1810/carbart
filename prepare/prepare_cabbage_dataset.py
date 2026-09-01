import os
import random
import shutil
from pathlib import Path


# ==========================================
# Kiểm tra file có phải ảnh không
# ==========================================
def is_image_file(filename):
    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    return Path(filename).suffix.lower() in image_exts


# ==========================================
# Tạo thư mục nếu chưa tồn tại
# ==========================================
def make_dir(path):
    os.makedirs(path, exist_ok=True)


# ==========================================
# Lấy toàn bộ ảnh trong thư mục
# recursive=True -> quét luôn thư mục con
# ==========================================
def collect_images(folder, recursive=False):
    image_paths = []

    if recursive:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if is_image_file(file):
                    image_paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder):
            full_path = os.path.join(folder, file)
            if os.path.isfile(full_path) and is_image_file(file):
                image_paths.append(full_path)

    return image_paths


# ==========================================
# Chia train / val / test
# ==========================================
def split_dataset(image_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    random.shuffle(image_list)

    total = len(image_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_images = image_list[:train_end]
    val_images = image_list[train_end:val_end]
    test_images = image_list[val_end:]

    return train_images, val_images, test_images


# ==========================================
# Copy ảnh sang thư mục mới
# prefix để đổi tên tránh trùng file
# ==========================================
def copy_images(image_list, dst_folder, prefix):
    make_dir(dst_folder)

    for idx, src_path in enumerate(image_list):
        ext = Path(src_path).suffix.lower()
        new_name = f"{prefix}_{idx:05d}{ext}"
        dst_path = os.path.join(dst_folder, new_name)
        shutil.copy2(src_path, dst_path)


# ==========================================
# Hàm chính
# ==========================================
def main():
    random.seed(42)

    # -------------------------------
    # Đường dẫn nguồn theo đúng cấu trúc bạn đưa
    # dataset_cabbage/data/cabbage
    # dataset_cabbage/data/non_cabbage
    # -------------------------------
    source_root = os.path.join("dataset_cabbage", "data")
    source_cabbage = os.path.join(source_root, "cabbage")
    source_non_cabbage = os.path.join(source_root, "non_cabbage")

    # -------------------------------
    # Thư mục đích để train
    # -------------------------------
    output_root = "dataset_cabbage_split"

    # Tỉ lệ chia
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # -------------------------------
    # Kiểm tra thư mục
    # -------------------------------
    if not os.path.exists(source_cabbage):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {source_cabbage}")

    if not os.path.exists(source_non_cabbage):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {source_non_cabbage}")

    # -------------------------------
    # Lấy ảnh
    # cabbage: quét cả thư mục con nếu có
    # non_cabbage: quét toàn bộ Bean, Broccoli...
    # -------------------------------
    cabbage_images = collect_images(source_cabbage, recursive=True)
    non_cabbage_images = collect_images(source_non_cabbage, recursive=True)

    print(f"Số ảnh cabbage: {len(cabbage_images)}")
    print(f"Số ảnh non_cabbage: {len(non_cabbage_images)}")

    if len(cabbage_images) == 0:
        raise ValueError("Không có ảnh nào trong dataset_cabbage/data/cabbage")

    if len(non_cabbage_images) == 0:
        raise ValueError("Không có ảnh nào trong dataset_cabbage/data/non_cabbage")

    # -------------------------------
    # Chia dữ liệu
    # -------------------------------
    cabbage_train, cabbage_val, cabbage_test = split_dataset(
        cabbage_images, train_ratio, val_ratio, test_ratio
    )

    non_train, non_val, non_test = split_dataset(
        non_cabbage_images, train_ratio, val_ratio, test_ratio
    )

    # -------------------------------
    # Tạo thư mục đầu ra
    # -------------------------------
    folders = [
        os.path.join(output_root, "train", "cabbage"),
        os.path.join(output_root, "train", "non_cabbage"),
        os.path.join(output_root, "val", "cabbage"),
        os.path.join(output_root, "val", "non_cabbage"),
        os.path.join(output_root, "test", "cabbage"),
        os.path.join(output_root, "test", "non_cabbage"),
    ]

    for folder in folders:
        make_dir(folder)

    # -------------------------------
    # Copy ảnh
    # -------------------------------
    copy_images(cabbage_train, os.path.join(output_root, "train", "cabbage"), "cabbage")
    copy_images(cabbage_val, os.path.join(output_root, "val", "cabbage"), "cabbage")
    copy_images(cabbage_test, os.path.join(output_root, "test", "cabbage"), "cabbage")

    copy_images(non_train, os.path.join(output_root, "train", "non_cabbage"), "noncabbage")
    copy_images(non_val, os.path.join(output_root, "val", "non_cabbage"), "noncabbage")
    copy_images(non_test, os.path.join(output_root, "test", "non_cabbage"), "noncabbage")

    print("\nĐã tạo xong dataset mới tại:", output_root)
    print("Cấu trúc đầu ra:")
    print("dataset_cabbage_split/")
    print("├── train/")
    print("│   ├── cabbage/")
    print("│   └── non_cabbage/")
    print("├── val/")
    print("│   ├── cabbage/")
    print("│   └── non_cabbage/")
    print("└── test/")
    print("    ├── cabbage/")
    print("    └── non_cabbage/")


if __name__ == "__main__":
    main()