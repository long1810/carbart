import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2


# =========================
# CẤU HÌNH
# =========================
SRC_DIR = Path("dataset_pest_original")

SRC_IMAGES_DIR = SRC_DIR / "images"
SRC_ANN_DIR = SRC_DIR / "annotations"

OUT_DIR = Path("dataset_pest_bbox_edges")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

SEED = 42

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

CLASS_NAMES = [
    "Ampelophaga",
    "army worm",
    "beet army worm",
    "cabbage army worm",
]

SOURCE_FOLDERS = [
    "Ampelophaga",
    "army worm",
    "beet army worm",
    "cabbage army worm",
    "Cabbage_test",
]

FOLDER_TO_CLASS = {
    "Ampelophaga": "Ampelophaga",
    "army worm": "army worm",
    "beet army worm": "beet army worm",
    "cabbage army worm": "cabbage army worm",
    "Cabbage_test": "cabbage army worm",
}

# Canny edge threshold
CANNY_LOW = 80
CANNY_HIGH = 160


def reset_output_dir():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for split in ["train", "val", "test"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "edges" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "annotations" / split).mkdir(parents=True, exist_ok=True)


def save_classes():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "classes.txt", "w", encoding="utf-8") as f:
        for name in CLASS_NAMES:
            f.write(name + "\n")


def find_images(folder):
    files = []

    if not folder.exists():
        return files

    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)

    return sorted(files)


def find_xml_for_image(folder_name, img_path):
    ann_folder = SRC_ANN_DIR / folder_name

    xml_path = ann_folder / f"{img_path.stem}.xml"

    if xml_path.exists():
        return xml_path

    matches = list(ann_folder.rglob(f"{img_path.stem}.xml"))

    if len(matches) > 0:
        return matches[0]

    return None


def parse_voc_xml(xml_path, default_class_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lines = []

    for obj in root.findall("object"):
        name_node = obj.find("name")

        if name_node is not None and name_node.text:
            class_name = name_node.text.strip()
        else:
            class_name = default_class_name

        if class_name == "Cabbage_test":
            class_name = "cabbage army worm"

        if class_name not in CLASS_NAMES:
            class_name = default_class_name

        if class_name not in CLASS_NAMES:
            print(f"Bỏ qua object vì class không hợp lệ: {class_name} trong {xml_path}")
            continue

        class_id = CLASS_NAMES.index(class_name)

        bndbox = obj.find("bndbox")

        if bndbox is None:
            print(f"Không có bndbox trong {xml_path}")
            continue

        try:
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
        except Exception:
            print(f"Lỗi đọc bndbox trong {xml_path}")
            continue

        if xmax <= xmin or ymax <= ymin:
            print(f"Box lỗi trong {xml_path}: {xmin}, {ymin}, {xmax}, {ymax}")
            continue

        lines.append(f"{class_id} {xmin} {ymin} {xmax} {ymax}")

    return lines


def safe_output_name(folder_name, img_path):
    safe_folder = folder_name.replace(" ", "_")
    return f"{safe_folder}_{img_path.name}"


def create_edge_image(src_img_path, dst_edge_path):
    image = cv2.imread(str(src_img_path))

    if image is None:
        print(f"Không đọc được ảnh để tạo edge: {src_img_path}")
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Làm mượt nhẹ để giảm nhiễu trước khi Canny
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    cv2.imwrite(str(dst_edge_path), edges)

    return True


def convert_one_sample(folder_name, img_path, xml_path, split, default_class_name):
    lines = parse_voc_xml(xml_path, default_class_name)

    if len(lines) == 0:
        print(f"Annotation rỗng, bỏ qua: {xml_path}")
        return False

    output_img_name = safe_output_name(folder_name, img_path)
    output_stem = Path(output_img_name).stem

    dst_img_path = OUT_DIR / "images" / split / output_img_name
    dst_edge_path = OUT_DIR / "edges" / split / f"{output_stem}.png"
    dst_ann_path = OUT_DIR / "annotations" / split / f"{output_stem}.txt"

    shutil.copy2(img_path, dst_img_path)

    ok_edge = create_edge_image(img_path, dst_edge_path)

    if not ok_edge:
        return False

    with open(dst_ann_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return True


def split_samples(samples):
    random.shuffle(samples)

    n = len(samples)

    if n == 0:
        return [], [], []

    if n == 1:
        return samples, [], []

    if n == 2:
        return samples[:1], samples[1:], []

    train_count = int(n * TRAIN_RATIO)
    val_count = int(n * VAL_RATIO)

    train_count = max(1, train_count)
    val_count = max(1, val_count)

    if train_count + val_count >= n:
        train_count = n - 2
        val_count = 1

    train_samples = samples[:train_count]
    val_samples = samples[train_count:train_count + val_count]
    test_samples = samples[train_count + val_count:]

    return train_samples, val_samples, test_samples


def collect_samples_from_folder(folder_name):
    img_folder = SRC_IMAGES_DIR / folder_name
    ann_folder = SRC_ANN_DIR / folder_name

    print(f"\nĐang xử lý folder: {folder_name}")
    print(f"Image folder: {img_folder}")
    print(f"Annotation folder: {ann_folder}")

    if not img_folder.exists():
        print(f"Không thấy thư mục ảnh: {img_folder}")
        return []

    if not ann_folder.exists():
        print(f"Không thấy thư mục annotation: {ann_folder}")
        return []

    default_class_name = FOLDER_TO_CLASS.get(folder_name)

    if default_class_name is None:
        print(f"Folder chưa được map class: {folder_name}")
        return []

    images = find_images(img_folder)

    print(f"Tìm thấy ảnh: {len(images)}")

    samples = []

    for img_path in images:
        xml_path = find_xml_for_image(folder_name, img_path)

        if xml_path is None:
            print(f"Thiếu XML cho ảnh: {img_path.name}")
            continue

        samples.append({
            "folder_name": folder_name,
            "img_path": img_path,
            "xml_path": xml_path,
            "default_class_name": default_class_name,
        })

    print(f"Cặp ảnh + XML hợp lệ: {len(samples)}")

    return samples


def copy_samples(samples, split):
    ok = 0

    for sample in samples:
        if convert_one_sample(
            folder_name=sample["folder_name"],
            img_path=sample["img_path"],
            xml_path=sample["xml_path"],
            split=split,
            default_class_name=sample["default_class_name"],
        ):
            ok += 1

    return ok


def count_output():
    print("\n===== KIỂM TRA OUTPUT =====")

    for split in ["train", "val", "test"]:
        img_dir = OUT_DIR / "images" / split
        edge_dir = OUT_DIR / "edges" / split
        ann_dir = OUT_DIR / "annotations" / split

        imgs = find_images(img_dir)
        edges = find_images(edge_dir)
        anns = list(ann_dir.glob("*.txt"))

        print(
            f"{split}: "
            f"images={len(imgs)}, "
            f"edges={len(edges)}, "
            f"annotations={len(anns)}"
        )


def main():
    random.seed(SEED)

    print("SRC_IMAGES_DIR:", SRC_IMAGES_DIR.resolve())
    print("SRC_ANN_DIR:", SRC_ANN_DIR.resolve())
    print("OUT_DIR:", OUT_DIR.resolve())

    if not SRC_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Không thấy thư mục ảnh: {SRC_IMAGES_DIR}")

    if not SRC_ANN_DIR.exists():
        raise FileNotFoundError(f"Không thấy thư mục annotation: {SRC_ANN_DIR}")

    reset_output_dir()
    save_classes()

    all_samples = []

    for folder_name in SOURCE_FOLDERS:
        samples = collect_samples_from_folder(folder_name)
        all_samples.extend(samples)

    print("\nTổng số cặp ảnh + XML hợp lệ:", len(all_samples))

    train_samples, val_samples, test_samples = split_samples(all_samples)

    train_ok = copy_samples(train_samples, "train")
    val_ok = copy_samples(val_samples, "val")
    test_ok = copy_samples(test_samples, "test")

    print("\n===== KẾT QUẢ CHIA DATASET =====")
    print(f"Train: {train_ok}")
    print(f"Val: {val_ok}")
    print(f"Test: {test_ok}")

    count_output()

    print("\nHoàn tất tạo dataset ảnh gốc + edge.")
    print(f"Dataset mới: {OUT_DIR}")


if __name__ == "__main__":
    main()