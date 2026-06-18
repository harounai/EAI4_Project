#!/usr/bin/env python3
"""
train_accessory.py
modified by Simone & Annalena after Mohamed & Vlad

Binary accessory classifier for Rock-Paper-Scissors project.
Input:  data_accessory/ folder with subfolders: absent/, present/
Output: artifacts/accessory_model.tflite  (int8 quantized)
        artifacts/accessory_model.keras
        artifacts/accessory_model_metrics.csv

Model:  MobileNetV3-Small fine-tuned on 224x224 RGB images.

Preprocessing: pixel values scaled to [0, 1] (MobileNetV3 standard).

IMPORTANT - this preprocessing contract must match C++ inference code in preprocess_accessory.h:
    float val = static_cast<float>(pixel) / 255.0f;

Class order is alphabetical (image_dataset_from_directory):
    0 = absent   (no accessory)
    1 = present  (accessory visible)
This must match model_classes.h: ACCESSORY_ABSENT=0, ACCESSORY_PRESENT=1
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

import pandas as pd



# ---------------------------------------------------------------------------
# Constants — must match model_classes.h
# ---------------------------------------------------------------------------

CLASS_NAMES = ["absent", "present"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = (224, 224)   # must match CaptureParameters in C++
BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train accessory detector and export to TFLite.")
    p.add_argument("--data-dir",        default="dataset")
    p.add_argument("--artifacts-dir",   default="artifacts")
    p.add_argument("--epochs",          type=int, default=20)
    p.add_argument("--unfreeze-epochs", type=int, default=10)
    p.add_argument("--val-split",       type=float, default=0.15)
    p.add_argument("--test-split",      type=float, default=0.15)
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading — tf.data pipeline, no numpy loading into RAM
# ---------------------------------------------------------------------------
ACCESSORY_MAP = {
    "absent": 0,
    "present": 1,
}

def load_dataset(data_dir: Path, seed: int):

    rows = []

    for gesture_dir in sorted(data_dir.iterdir()):

        if not gesture_dir.is_dir():
            continue

        for accessory_dir in sorted(gesture_dir.iterdir()):

            if not accessory_dir.is_dir():
                continue

            accessory_label = (
                "present"
                if accessory_dir.name == "with"
                else "absent"
            )

            for image_path in sorted(accessory_dir.glob("*.bmp")):

                rows.append({
                    "path": str(image_path),
                    "label": ACCESSORY_MAP[accessory_label],
                })

    df = pd.DataFrame(rows)

    print(f"Loaded {len(df)} images")

    if len(df) == 0:
        raise ValueError(
            f"No BMP images found in {data_dir}"
        )

    paths = df["path"].values
    labels = df["label"].values

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    def load_image(path, label):

        image = tf.io.read_file(path)
        image = tf.image.decode_bmp(
            image,
            channels=3
        )

        image = tf.image.resize(
            image,
            IMAGE_SIZE
        )

        return image, label

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.shuffle(
        len(df),
        seed=seed
    )

    ds = ds.batch(
        BATCH_SIZE
    )

    return ds


def make_pipelines(full_ds):
    """Split dataset into train/val/test and apply preprocessing."""
    total = sum(1 for _ in full_ds)
    test_batches = max(1, int(total * 0.15))
    val_batches  = max(1, int(total * 0.15))

    test_ds  = full_ds.take(test_batches)
    val_ds   = full_ds.skip(test_batches).take(val_batches)
    train_ds = full_ds.skip(test_batches + val_batches)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.1),
    ])

    def preprocess(images, labels):
        images = tf.cast(images, tf.float32)
        images = images / 255.0
        return images, labels

    def preprocess_and_augment(images, labels):
        images = tf.cast(images, tf.float32)
        images = augment(images, training=True)
        images = images / 255.0
        return images, labels

    train_ds = train_ds.map(preprocess_and_augment, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_model(num_classes: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image_input")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="accessory_mobilenetv3small")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_top_layers(model: tf.keras.Model, n_unfreeze: int = 20) -> None:
    base = model.get_layer("MobileNetV3Small")
    base.trainable = True
    for layer in base.layers[:-n_unfreeze]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Unfroze last {n_unfreeze} backbone layers for fine-tuning.")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def make_callbacks() -> list:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------

def export_tflite(model: tf.keras.Model, artifacts_dir: Path,
                  model_name: str, calib_ds) -> Path:
    keras_path      = artifacts_dir / f"{model_name}.keras"
    saved_model_dir = artifacts_dir / f"{model_name}_saved_model"
    tflite_path     = artifacts_dir / f"{model_name}.tflite"

    model.save(str(keras_path))

    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    model.export(str(saved_model_dir))

    def representative_dataset():
        for images, _ in calib_ds.take(50):
            for i in range(len(images)):
                yield [tf.expand_dims(images[i], 0)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_bytes = converter.convert()
    tflite_path.write_bytes(tflite_bytes)

    print(f"\nExported: {tflite_path}  ({len(tflite_bytes)/1024:.1f} KB)")
    return tflite_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    data_dir      = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("--- Loading data ---")
    full_ds = load_dataset(data_dir, args.seed)

    print("\n--- Building pipelines ---")
    train_ds, val_ds, test_ds = make_pipelines(full_ds)

    print("\n--- Building model ---")
    model = make_model(NUM_CLASSES)
    model.summary()

    print("\n--- Phase 1: head only ---")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              callbacks=make_callbacks(), verbose=2)

    print("\n--- Phase 2: fine-tune backbone ---")
    unfreeze_top_layers(model, n_unfreeze=20)
    model.fit(train_ds, validation_data=val_ds, epochs=args.unfreeze_epochs,
              callbacks=make_callbacks(), verbose=2)

    print("\n--- Evaluating ---")
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test loss: {loss:.4f}  Test accuracy: {acc*100:.2f}%")

    if acc < 0.90:
        print("\nWARNING: accuracy below 90%. Consider collecting more data.")

    print("\n--- Exporting TFLite (int8) ---")
    tflite_path = export_tflite(model, artifacts_dir, "accessory_model", train_ds)

    print("\n--- Saving metrics ---")
    metrics_path = artifacts_dir / "accessory_model_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows([
            {"metric": "test_accuracy_keras", "value": f"{acc:.4f}"},
            {"metric": "test_loss",           "value": f"{loss:.4f}"},
            {"metric": "tflite_size_kb",      "value": f"{tflite_path.stat().st_size/1024:.1f}"},
            {"metric": "class_0",             "value": CLASS_NAMES[0]},
            {"metric": "class_1",             "value": CLASS_NAMES[1]},
        ])

    print(f"\n=== Done ===")
    print(f"Model:    {tflite_path}")
    print(f"Size:     {tflite_path.stat().st_size/1024:.1f} KB")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"\nClass mapping:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {i} = {cls}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())