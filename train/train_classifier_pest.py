
import os

# Giới hạn số thread CPU TensorFlow dùng
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
# =========================
# CẤU HÌNH CỐ ĐỊNH
# =========================
DATA_DIR = Path("dataset_pest_split")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = Path("runs/pest_mobilenet")

MODEL_NAME = "mobilenetv2"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
SEED = 42


def build_mobilenetv2(num_classes):
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    return model


def load_dataset():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    return train_ds, val_ds, test_ds


def save_classes(class_names):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classes_path = OUTPUT_DIR / "classes.txt"

    with open(classes_path, "w", encoding="utf-8") as f:
        for class_name in class_names:
            f.write(class_name + "\n")

    print(f"Đã lưu classes tại: {classes_path}")


def train_model():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = load_dataset()

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print("Class names:", class_names)
    print("Số class:", num_classes)

    save_classes(class_names)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    if MODEL_NAME == "mobilenetv2":
        model = build_mobilenetv2(num_classes)
    else:
        raise ValueError("Hiện tại chỉ hỗ trợ MobileNetV2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    best_model_path = OUTPUT_DIR / "best_model.keras"
    final_model_path = OUTPUT_DIR / "final_model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(final_model_path)

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    print("Train xong.")
    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_model_path}")


def main():
    train_model()


if __name__ == "__main__":
    main()