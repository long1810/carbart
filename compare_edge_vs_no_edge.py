import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv


# =========================
# CẤU HÌNH MODEL KHÔNG EDGE
# =========================
RGB_MODEL_PATH = Path("runs/pest_bbox_mobilenet/best_model.keras")
RGB_CLASSES_PATH = Path("runs/pest_bbox_mobilenet/classes.txt")
RGB_TEST_IMG_DIR = Path("dataset_pest_bbox/images/test")
RGB_TEST_ANN_DIR = Path("dataset_pest_bbox/annotations/test")


# =========================
# CẤU HÌNH MODEL CÓ EDGE
# =========================
EDGE_MODEL_PATH = Path("runs/pest_bbox_mobilenet_edges/best_model.keras")
EDGE_CLASSES_PATH = Path("runs/pest_bbox_mobilenet_edges/classes.txt")
EDGE_TEST_IMG_DIR = Path("dataset_pest_bbox_edges/images/test")
EDGE_TEST_ANN_DIR = Path("dataset_pest_bbox_edges/annotations/test")


# =========================
# OUTPUT
# =========================
OUTPUT_DIR = Path("compare_results_edge_vs_no_edge")

IMG_SIZE = 224
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

CANNY_LOW = 80
CANNY_HIGH = 160


def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def get_image_files(img_dir):
    files = []

    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)

    return sorted(files)


def read_annotation(txt_path, img_w, img_h):
    """
    Đọc annotation TXT:
    class_id xmin ymin xmax ymax
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) == 0:
        return None

    # Bản hiện tại lấy object đầu tiên
    parts = lines[0].split()

    if len(parts) != 5:
        return None

    class_id = int(parts[0])

    x1 = float(parts[1])
    y1 = float(parts[2])
    x2 = float(parts[3])
    y2 = float(parts[4])

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    box_pixel = np.array([x1, y1, x2, y2], dtype=np.float32)

    box_norm = np.array(
        [
            x1 / img_w,
            y1 / img_h,
            x2 / img_w,
            y2 / img_h,
        ],
        dtype=np.float32
    )

    return class_id, box_pixel, box_norm


def box_iou(box_a, box_b):
    """
    Tính IoU giữa 2 box dạng pixel:
    [x1, y1, x2, y2]
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return float(inter_area / union_area)


def create_edge_for_model(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    edges_resized = cv2.resize(edges, (IMG_SIZE, IMG_SIZE))
    edges_resized = edges_resized.astype(np.float32) / 255.0
    edges_resized = np.expand_dims(edges_resized, axis=-1)

    return edges_resized


def predict_rgb_model(model, image_bgr):
    orig_h, orig_w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    image_input = np.expand_dims(image_resized.astype(np.float32), axis=0)

    outputs = model.predict(image_input, verbose=0)

    if isinstance(outputs, dict):
        class_pred = outputs["class_output"][0]
        box_pred = outputs["box_output"][0]
    else:
        class_pred = outputs[0][0]
        box_pred = outputs[1][0]

    pred_class_id = int(np.argmax(class_pred))
    confidence = float(class_pred[pred_class_id])

    x_min, y_min, x_max, y_max = box_pred

    pred_box_pixel = np.array(
        [
            x_min * orig_w,
            y_min * orig_h,
            x_max * orig_w,
            y_max * orig_h,
        ],
        dtype=np.float32
    )

    pred_box_pixel[0] = max(0, min(pred_box_pixel[0], orig_w - 1))
    pred_box_pixel[1] = max(0, min(pred_box_pixel[1], orig_h - 1))
    pred_box_pixel[2] = max(0, min(pred_box_pixel[2], orig_w - 1))
    pred_box_pixel[3] = max(0, min(pred_box_pixel[3], orig_h - 1))

    pred_box_norm = np.array(
        [
            pred_box_pixel[0] / orig_w,
            pred_box_pixel[1] / orig_h,
            pred_box_pixel[2] / orig_w,
            pred_box_pixel[3] / orig_h,
        ],
        dtype=np.float32
    )

    return pred_class_id, confidence, pred_box_pixel, pred_box_norm


def predict_edge_model(model, image_bgr):
    orig_h, orig_w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    image_input = np.expand_dims(image_resized.astype(np.float32), axis=0)

    edge_input = create_edge_for_model(image_bgr)
    edge_input = np.expand_dims(edge_input, axis=0)

    outputs = model.predict(
        {
            "image_input": image_input,
            "edge_input": edge_input,
        },
        verbose=0
    )

    if isinstance(outputs, dict):
        class_pred = outputs["class_output"][0]
        box_pred = outputs["box_output"][0]
    else:
        class_pred = outputs[0][0]
        box_pred = outputs[1][0]

    pred_class_id = int(np.argmax(class_pred))
    confidence = float(class_pred[pred_class_id])

    x_min, y_min, x_max, y_max = box_pred

    pred_box_pixel = np.array(
        [
            x_min * orig_w,
            y_min * orig_h,
            x_max * orig_w,
            y_max * orig_h,
        ],
        dtype=np.float32
    )

    pred_box_pixel[0] = max(0, min(pred_box_pixel[0], orig_w - 1))
    pred_box_pixel[1] = max(0, min(pred_box_pixel[1], orig_h - 1))
    pred_box_pixel[2] = max(0, min(pred_box_pixel[2], orig_w - 1))
    pred_box_pixel[3] = max(0, min(pred_box_pixel[3], orig_h - 1))

    pred_box_norm = np.array(
        [
            pred_box_pixel[0] / orig_w,
            pred_box_pixel[1] / orig_h,
            pred_box_pixel[2] / orig_w,
            pred_box_pixel[3] / orig_h,
        ],
        dtype=np.float32
    )

    return pred_class_id, confidence, pred_box_pixel, pred_box_norm


def evaluate_model(
    model_name,
    model_path,
    classes_path,
    img_dir,
    ann_dir,
    use_edge=False
):
    print(f"\n===== ĐÁNH GIÁ MODEL: {model_name} =====")

    if not model_path.exists():
        raise FileNotFoundError(f"Không thấy model: {model_path}")

    if not classes_path.exists():
        raise FileNotFoundError(f"Không thấy classes.txt: {classes_path}")

    if not img_dir.exists():
        raise FileNotFoundError(f"Không thấy thư mục ảnh test: {img_dir}")

    if not ann_dir.exists():
        raise FileNotFoundError(f"Không thấy thư mục annotation test: {ann_dir}")

    model = tf.keras.models.load_model(model_path)
    class_names = load_classes(classes_path)

    image_files = get_image_files(img_dir)

    print("Số ảnh test:", len(image_files))
    print("Classes:", class_names)

    total = 0
    correct_class = 0

    iou_list = []
    mae_list = []
    confidence_list = []
    iou50_correct = 0

    detail_rows = []

    for idx, image_path in enumerate(image_files, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Bỏ qua ảnh lỗi: {image_path}")
            continue

        h, w = image.shape[:2]

        ann_path = ann_dir / f"{image_path.stem}.txt"

        if not ann_path.exists():
            print(f"Thiếu annotation: {ann_path.name}")
            continue

        ann = read_annotation(ann_path, w, h)

        if ann is None:
            print(f"Annotation lỗi: {ann_path.name}")
            continue

        true_class_id, true_box_pixel, true_box_norm = ann

        if use_edge:
            pred_class_id, confidence, pred_box_pixel, pred_box_norm = predict_edge_model(model, image)
        else:
            pred_class_id, confidence, pred_box_pixel, pred_box_norm = predict_rgb_model(model, image)

        iou = box_iou(true_box_pixel, pred_box_pixel)
        mae = float(np.mean(np.abs(true_box_norm - pred_box_norm)))

        total += 1

        if pred_class_id == true_class_id:
            correct_class += 1

        if iou >= 0.5:
            iou50_correct += 1

        iou_list.append(iou)
        mae_list.append(mae)
        confidence_list.append(confidence)

        detail_rows.append({
            "image": image_path.name,
            "true_class_id": true_class_id,
            "pred_class_id": pred_class_id,
            "confidence": confidence,
            "iou": iou,
            "box_mae": mae,
            "true_box": true_box_pixel.tolist(),
            "pred_box": pred_box_pixel.tolist(),
        })

    if total == 0:
        raise ValueError(f"Không có mẫu test hợp lệ cho model: {model_name}")

    result = {
        "model_name": model_name,
        "total": total,
        "class_accuracy": correct_class / total,
        "mean_iou": float(np.mean(iou_list)),
        "iou_50_accuracy": iou50_correct / total,
        "box_mae": float(np.mean(mae_list)),
        "mean_confidence": float(np.mean(confidence_list)),
        "details": detail_rows,
    }

    print(f"Tổng mẫu hợp lệ: {result['total']}")
    print(f"Class Accuracy: {result['class_accuracy'] * 100:.2f}%")
    print(f"Mean IoU: {result['mean_iou']:.4f}")
    print(f"IoU@0.5: {result['iou_50_accuracy'] * 100:.2f}%")
    print(f"Box MAE: {result['box_mae']:.4f}")
    print(f"Mean Confidence: {result['mean_confidence'] * 100:.2f}%")

    return result


def save_summary_csv(results, output_dir):
    csv_path = output_dir / "summary_compare.csv"

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "model",
            "total",
            "class_accuracy",
            "mean_iou",
            "iou_50_accuracy",
            "box_mae",
            "mean_confidence"
        ])

        for r in results:
            writer.writerow([
                r["model_name"],
                r["total"],
                r["class_accuracy"],
                r["mean_iou"],
                r["iou_50_accuracy"],
                r["box_mae"],
                r["mean_confidence"]
            ])

    print(f"Đã lưu CSV tổng hợp: {csv_path}")


def save_detail_csv(results, output_dir):
    csv_path = output_dir / "details_compare.csv"

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "model",
            "image",
            "true_class_id",
            "pred_class_id",
            "confidence",
            "iou",
            "box_mae",
            "true_box",
            "pred_box"
        ])

        for r in results:
            for row in r["details"]:
                writer.writerow([
                    r["model_name"],
                    row["image"],
                    row["true_class_id"],
                    row["pred_class_id"],
                    row["confidence"],
                    row["iou"],
                    row["box_mae"],
                    row["true_box"],
                    row["pred_box"],
                ])

    print(f"Đã lưu CSV chi tiết: {csv_path}")


def plot_compare(results, output_dir):
    names = [r["model_name"] for r in results]

    metrics = [
        ("class_accuracy", "Class Accuracy", "compare_class_accuracy.png"),
        ("mean_iou", "Mean IoU", "compare_mean_iou.png"),
        ("iou_50_accuracy", "IoU@0.5 Accuracy", "compare_iou50.png"),
        ("box_mae", "Box MAE", "compare_box_mae.png"),
        ("mean_confidence", "Mean Confidence", "compare_confidence.png"),
    ]

    for metric_key, title, filename in metrics:
        values = [r[metric_key] for r in results]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(names, values)

        plt.title(title)
        plt.ylabel(metric_key)
        plt.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{value:.4f}",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()

        save_path = output_dir / filename
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"Đã lưu biểu đồ: {save_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    result_rgb = evaluate_model(
        model_name="RGB only",
        model_path=RGB_MODEL_PATH,
        classes_path=RGB_CLASSES_PATH,
        img_dir=RGB_TEST_IMG_DIR,
        ann_dir=RGB_TEST_ANN_DIR,
        use_edge=False
    )

    results.append(result_rgb)

    result_edge = evaluate_model(
        model_name="RGB + Edge",
        model_path=EDGE_MODEL_PATH,
        classes_path=EDGE_CLASSES_PATH,
        img_dir=EDGE_TEST_IMG_DIR,
        ann_dir=EDGE_TEST_ANN_DIR,
        use_edge=True
    )

    results.append(result_edge)

    save_summary_csv(results, OUTPUT_DIR)
    save_detail_csv(results, OUTPUT_DIR)
    plot_compare(results, OUTPUT_DIR)

    print("\n===== KẾT LUẬN NHANH =====")

    if result_edge["mean_iou"] > result_rgb["mean_iou"]:
        print("Model RGB + Edge có Mean IoU cao hơn, tức là box dự đoán khớp vùng sâu tốt hơn.")
    else:
        print("Model RGB only có Mean IoU cao hơn hoặc tương đương, Edge chưa cải thiện vị trí box.")

    if result_edge["class_accuracy"] > result_rgb["class_accuracy"]:
        print("Model RGB + Edge có độ chính xác phân loại cao hơn.")
    else:
        print("Model RGB only có độ chính xác phân loại cao hơn hoặc tương đương.")

    if result_edge["box_mae"] < result_rgb["box_mae"]:
        print("Model RGB + Edge có Box MAE thấp hơn, tức là sai số tọa độ nhỏ hơn.")
    else:
        print("Model RGB only có Box MAE thấp hơn hoặc tương đương.")


if __name__ == "__main__":
    main()