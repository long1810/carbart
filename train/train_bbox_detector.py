import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
# =========================
# GIỚI HẠN CPU KHOẢNG 80%
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
DATA_DIR = Path("dataset_pest_bbox")

TRAIN_IMG_DIR = DATA_DIR / "images" / "train"
VAL_IMG_DIR = DATA_DIR / "images" / "val"

TRAIN_ANN_DIR = DATA_DIR / "annotations" / "train"
VAL_ANN_DIR = DATA_DIR / "annotations" / "val"

CLASSES_PATH = DATA_DIR / "classes.txt"

OUTPUT_DIR = Path("runs/pest_bbox_mobilenet")

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4


def load_classes():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def load_samples(img_dir, ann_dir):
    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    samples = []

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue

        ann_path = ann_dir / f"{img_path.stem}.txt"

        if not ann_path.exists():
            print(f"Bỏ qua vì thiếu annotation: {img_path.name}")
            continue

        samples.append((img_path, ann_path))

    return samples


def read_annotation(txt_path, img_w, img_h):
    """
    Format annotation:
    class_id x_min y_min x_max y_max

    Bản này lấy box đầu tiên trong file.
    Phù hợp khi mỗi ảnh có 1 con sâu chính.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) == 0:
        raise ValueError(f"Annotation rỗng: {txt_path}")

    parts = lines[0].split()

    if len(parts) != 5:
        raise ValueError(f"Sai format annotation: {txt_path}")

    class_id = int(parts[0])

    x_min = float(parts[1])
    y_min = float(parts[2])
    x_max = float(parts[3])
    y_max = float(parts[4])

    # Chặn tọa độ trong kích thước ảnh
    x_min = max(0, min(x_min, img_w - 1))
    y_min = max(0, min(y_min, img_h - 1))
    x_max = max(0, min(x_max, img_w - 1))
    y_max = max(0, min(y_max, img_h - 1))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Box không hợp lệ: {txt_path}")

    # Normalize về 0..1
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
            class_targets = []
            box_targets = []

            for img_path, ann_path in batch_samples:
                img = cv2.imread(str(img_path))

                if img is None:
                    print(f"Không đọc được ảnh: {img_path}")
                    continue

                h, w = img.shape[:2]

                try:
                    class_id, box = read_annotation(ann_path, w, h)
                except Exception as e:
                    print(e)
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

                images.append(img_resized.astype(np.float32))

                class_one_hot = tf.keras.utils.to_categorical(
                    class_id,
                    num_classes=num_classes
                )

                class_targets.append(class_one_hot)
                box_targets.append(box)

            if len(images) == 0:
                continue

            images = np.array(images, dtype=np.float32)
            class_targets = np.array(class_targets, dtype=np.float32)
            box_targets = np.array(box_targets, dtype=np.float32)

            yield images, {
                "class_output": class_targets,
                "box_output": box_targets
            }


def build_bbox_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    class_output = layers.Dense(
        num_classes,
        activation="softmax",
        name="class_output"
    )(x)

    box_output = layers.Dense(
        4,
        activation="sigmoid",
        name="box_output"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs={
            "class_output": class_output,
            "box_output": box_output
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

    train_samples = load_samples(TRAIN_IMG_DIR, TRAIN_ANN_DIR)
    val_samples = load_samples(VAL_IMG_DIR, VAL_ANN_DIR)

    print("Train samples:", len(train_samples))
    print("Val samples:", len(val_samples))

    if len(train_samples) == 0:
        raise ValueError("Không có ảnh train hợp lệ.")

    if len(val_samples) == 0:
        raise ValueError("Không có ảnh val hợp lệ.")

    train_gen = data_generator(train_samples, num_classes)
    val_gen = data_generator(val_samples, num_classes)

    steps_per_epoch = max(1, len(train_samples) // BATCH_SIZE)
    validation_steps = max(1, len(val_samples) // BATCH_SIZE)

    model = build_bbox_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss={
            "class_output": "categorical_crossentropy",
            "box_output": "mse"
        },
        loss_weights={
            "class_output": 1.0,
            "box_output": 5.0
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
            patience=7,
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