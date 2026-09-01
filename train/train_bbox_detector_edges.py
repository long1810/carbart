import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
# =========================
# GIỚI HẠN CPU + GIẢM LOG
# =========================
cpu_count = os.cpu_count() or 4
cpu_80 = max(1, int(cpu_count * 0.8))
inter_threads = max(1, cpu_80 // 2)

os.environ["OMP_NUM_THREADS"] = str(cpu_80)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(cpu_80)
os.environ["TF_NUM_INTEROP_THREADS"] = str(inter_threads)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

tf.config.threading.set_intra_op_parallelism_threads(cpu_80)
tf.config.threading.set_inter_op_parallelism_threads(inter_threads)

print(f"CPU threads: {cpu_80}/{cpu_count} (~80%)")


# =========================
# CẤU HÌNH
# =========================
DATA_DIR = Path("dataset_pest_bbox_edges")

TRAIN_IMG_DIR = DATA_DIR / "images" / "train"
VAL_IMG_DIR = DATA_DIR / "images" / "val"

TRAIN_EDGE_DIR = DATA_DIR / "edges" / "train"
VAL_EDGE_DIR = DATA_DIR / "edges" / "val"

TRAIN_ANN_DIR = DATA_DIR / "annotations" / "train"
VAL_ANN_DIR = DATA_DIR / "annotations" / "val"

CLASSES_PATH = DATA_DIR / "classes.txt"

OUTPUT_DIR = Path("runs/pest_bbox_mobilenet_edges")

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def load_samples(img_dir, edge_dir, ann_dir):
    samples = []

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        stem = img_path.stem

        edge_path = edge_dir / f"{stem}.png"
        ann_path = ann_dir / f"{stem}.txt"

        if not edge_path.exists():
            print(f"Bỏ qua vì thiếu edge: {edge_path.name}")
            continue

        if not ann_path.exists():
            print(f"Bỏ qua vì thiếu annotation: {ann_path.name}")
            continue

        samples.append((img_path, edge_path, ann_path))

    return samples


def read_annotation(txt_path, img_w, img_h):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) == 0:
        raise ValueError(f"Annotation rỗng: {txt_path}")

    # Bản đơn giản: lấy object đầu tiên
    parts = lines[0].split()

    if len(parts) != 5:
        raise ValueError(f"Sai format annotation: {txt_path}")

    class_id = int(parts[0])

    x_min = float(parts[1])
    y_min = float(parts[2])
    x_max = float(parts[3])
    y_max = float(parts[4])

    x_min = max(0, min(x_min, img_w - 1))
    y_min = max(0, min(y_min, img_h - 1))
    x_max = max(0, min(x_max, img_w - 1))
    y_max = max(0, min(y_max, img_h - 1))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Box không hợp lệ: {txt_path}")

    box = [
        x_min / img_w,
        y_min / img_h,
        x_max / img_w,
        y_max / img_h,
    ]

    return class_id, box


def data_generator(samples, num_classes):
    while True:
        np.random.shuffle(samples)

        for i in range(0, len(samples), BATCH_SIZE):
            batch_samples = samples[i:i + BATCH_SIZE]

            images = []
            edges = []
            class_targets = []
            box_targets = []

            for img_path, edge_path, ann_path in batch_samples:
                img = cv2.imread(str(img_path))
                edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"Không đọc được ảnh: {img_path}")
                    continue

                if edge is None:
                    print(f"Không đọc được edge: {edge_path}")
                    continue

                h, w = img.shape[:2]

                try:
                    class_id, box = read_annotation(ann_path, w, h)
                except Exception as e:
                    print(e)
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

                edge_resized = cv2.resize(edge, (IMG_SIZE, IMG_SIZE))
                edge_resized = np.expand_dims(edge_resized, axis=-1)

                images.append(img_resized.astype(np.float32))

                # Edge đưa về 0..1
                edges.append(edge_resized.astype(np.float32) / 255.0)

                class_one_hot = tf.keras.utils.to_categorical(
                    class_id,
                    num_classes=num_classes
                )

                class_targets.append(class_one_hot)
                box_targets.append(box)

            if len(images) == 0:
                continue

            images = np.array(images, dtype=np.float32)
            edges = np.array(edges, dtype=np.float32)
            class_targets = np.array(class_targets, dtype=np.float32)
            box_targets = np.array(box_targets, dtype=np.float32)

            yield {
                "image_input": images,
                "edge_input": edges
            }, {
                "class_output": class_targets,
                "box_output": box_targets
            }


# def build_model(num_classes):
#     # =========================
#     # NHÁNH ẢNH GỐC RGB
#     # =========================
#     image_input = layers.Input(
#         shape=(IMG_SIZE, IMG_SIZE, 3),
#         name="image_input"
#     )

#     x = tf.keras.applications.mobilenet_v2.preprocess_input(image_input)

#     base_model = tf.keras.applications.MobileNetV2(
#         input_shape=(IMG_SIZE, IMG_SIZE, 3),
#         include_top=False,
#         weights="imagenet"
#     )

#     base_model.trainable = False

#     x = base_model(x, training=False)
#     x = layers.GlobalAveragePooling2D(name="rgb_gap")(x)

#     # =========================
#     # NHÁNH EDGE
#     # =========================
#     edge_input = layers.Input(
#         shape=(IMG_SIZE, IMG_SIZE, 1),
#         name="edge_input"
#     )

#     e = layers.Conv2D(16, 3, padding="same", activation="relu")(edge_input)
#     e = layers.MaxPooling2D()(e)

#     e = layers.Conv2D(32, 3, padding="same", activation="relu")(e)
#     e = layers.MaxPooling2D()(e)

#     e = layers.Conv2D(64, 3, padding="same", activation="relu")(e)
#     e = layers.MaxPooling2D()(e)

#     e = layers.GlobalAveragePooling2D(name="edge_gap")(e)

#     # =========================
#     # GHÉP ĐẶC TRƯNG
#     # =========================
#     combined = layers.Concatenate(name="concat_rgb_edge")([x, e])

#     combined = layers.Dense(256, activation="relu")(combined)
#     combined = layers.Dropout(0.3)(combined)

#     class_output = layers.Dense(
#         num_classes,
#         activation="softmax",
#         name="class_output"
#     )(combined)

#     box_output = layers.Dense(
#         4,
#         activation="sigmoid",
#         name="box_output"
#     )(combined)

#     model = models.Model(
#         inputs={
#             "image_input": image_input,
#             "edge_input": edge_input,
#         },
#         outputs={
#             "class_output": class_output,
#             "box_output": box_output,
#         }
#     )

#     return model
def build_model(num_classes):
    # =========================
    # INPUT ẢNH GỐC RGB
    # =========================
    image_input = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="image_input"
    )

    x = tf.keras.applications.mobilenet_v2.preprocess_input(image_input)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    # Giai đoạn đầu đóng băng MobileNetV2 cho ổn định
    base_model.trainable = False

    rgb_feat = base_model(x, training=False)
    # Với IMG_SIZE=224, rgb_feat thường là 7x7x1280

    # =========================
    # INPUT EDGE
    # =========================
    edge_input = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 1),
        name="edge_input"
    )

    e = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(edge_input)   # 112x112
    e = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(e)            # 56x56
    e = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(e)            # 28x28
    e = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(e)           # 14x14
    e = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(e)           # 7x7

    # =========================
    # GHÉP FEATURE MAP RGB + EDGE
    # =========================
    combined = layers.Concatenate(name="concat_rgb_edge_feature")([rgb_feat, e])

    # Giữ thông tin không gian bằng Conv2D thay vì GlobalAveragePooling quá sớm
    combined = layers.Conv2D(256, 3, padding="same", activation="relu")(combined)
    combined = layers.BatchNormalization()(combined)

    combined = layers.Conv2D(128, 3, padding="same", activation="relu")(combined)
    combined = layers.BatchNormalization()(combined)

    combined = layers.Dropout(0.25)(combined)

    # Flatten giữ thông tin vị trí tốt hơn GAP cho bài toán box đơn giản
    combined = layers.Flatten()(combined)

    combined = layers.Dense(256, activation="relu")(combined)
    combined = layers.Dropout(0.3)(combined)

    # =========================
    # OUTPUT CLASS
    # =========================
    class_output = layers.Dense(
        num_classes,
        activation="softmax",
        name="class_output"
    )(combined)

    # =========================
    # OUTPUT BOX
    # =========================
    box_output = layers.Dense(
        4,
        activation="sigmoid",
        name="box_output"
    )(combined)

    model = models.Model(
        inputs={
            "image_input": image_input,
            "edge_input": edge_input,
        },
        outputs={
            "class_output": class_output,
            "box_output": box_output,
        }
    )

    return model
def plot_training_history(history, output_dir):
    """
    Lưu biểu đồ quá trình train:
    - total loss
    - class loss
    - box loss
    - class accuracy
    - box mae
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_dict = history.history

    # Lưu history ra JSON để sau này đưa vào báo cáo
    history_json_path = output_dir / "training_history.json"
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu history: {history_json_path}")

    def save_plot(keys, title, ylabel, filename):
        plt.figure(figsize=(10, 6))

        has_data = False

        for key in keys:
            if key in history_dict:
                plt.plot(history_dict[key], label=key)
                has_data = True

        if not has_data:
            print(f"Bỏ qua biểu đồ {filename} vì không có key phù hợp.")
            return

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = output_dir / filename
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"Đã lưu biểu đồ: {save_path}")

    # 1. Biểu đồ tổng loss
    save_plot(
        keys=["loss", "val_loss"],
        title="Training Loss and Validation Loss",
        ylabel="Loss",
        filename="loss_total.png"
    )

    # 2. Biểu đồ loss phân loại
    save_plot(
        keys=["class_output_loss", "val_class_output_loss"],
        title="Class Output Loss",
        ylabel="Categorical Crossentropy",
        filename="loss_class_output.png"
    )

    # 3. Biểu đồ loss bounding box
    save_plot(
        keys=["box_output_loss", "val_box_output_loss"],
        title="Bounding Box Output Loss",
        ylabel="Box Loss",
        filename="loss_box_output.png"
    )

    # 4. Biểu đồ accuracy phân loại
    save_plot(
        keys=["class_output_accuracy", "val_class_output_accuracy"],
        title="Class Output Accuracy",
        ylabel="Accuracy",
        filename="accuracy_class_output.png"
    )

    # 5. Biểu đồ MAE của bounding box
    save_plot(
        keys=["box_output_mae", "val_box_output_mae"],
        title="Bounding Box MAE",
        ylabel="MAE",
        filename="mae_box_output.png"
    )
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_IMG_DIR.exists():
        raise FileNotFoundError(f"Không thấy thư mục train ảnh: {TRAIN_IMG_DIR}")

    if not VAL_IMG_DIR.exists():
        raise FileNotFoundError(f"Không thấy thư mục val ảnh: {VAL_IMG_DIR}")

    class_names = load_classes()
    num_classes = len(class_names)

    print("Classes:", class_names)

    train_samples = load_samples(TRAIN_IMG_DIR, TRAIN_EDGE_DIR, TRAIN_ANN_DIR)
    val_samples = load_samples(VAL_IMG_DIR, VAL_EDGE_DIR, VAL_ANN_DIR)

    print("Train samples:", len(train_samples))
    print("Val samples:", len(val_samples))

    if len(train_samples) == 0:
        raise ValueError("Không có ảnh train hợp lệ.")

    if len(val_samples) == 0:
        print("Không có ảnh val, tạm dùng train làm val.")
        val_samples = train_samples

    train_gen = data_generator(train_samples, num_classes)
    val_gen = data_generator(val_samples, num_classes)

    steps_per_epoch = max(1, len(train_samples) // BATCH_SIZE)
    validation_steps = max(1, len(val_samples) // BATCH_SIZE)

    model = build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss={
            "class_output": "categorical_crossentropy",
            "box_output": "mse"
        },
        loss_weights={
            "class_output": 1.0,
            "box_output": 7.0
        },
        metrics={
            "class_output": ["accuracy"],
            "box_output": ["mae"]
        }
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=OUTPUT_DIR / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )
    ]

    history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks
    )
    plot_training_history(history, OUTPUT_DIR)

    model.save(OUTPUT_DIR / "final_model.keras")

    with open(OUTPUT_DIR / "classes.txt", "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    print("Train xong.")
    print("Best model:", OUTPUT_DIR / "best_model.keras")
    print("Final model:", OUTPUT_DIR / "final_model.keras")
    print("Classes:", OUTPUT_DIR / "classes.txt")


if __name__ == "__main__":
    main()