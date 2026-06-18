#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import struct
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

DISTILL_EPOCHS = 3

def require_tfmot():
    try:
        import tf_keras  # noqa: F401
        from tensorflow_model_optimization.clustering import keras as clustering
    except ImportError as exc:
        raise RuntimeError(
            "TFMOT clustering requires compatible tensorflow, tf-keras, "
            "tensorflow-model-optimization, and numpy versions."
        ) from exc
    return clustering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MNIST baseline/compression variants, one student KD model, one K-Means example, and one final INT8 student."
    )
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=1)
    parser.add_argument("--cluster-finetune-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--conv-filters", type=int, default=16)
    parser.add_argument("--dense-units", type=int, default=64)
    parser.add_argument("--synapse-prune-ratio", type=float, default=0.70)
    parser.add_argument("--neuron-prune-ratio", type=float, default=0.50)
    parser.add_argument("--channel-prune-ratio", type=float, default=0.50)
    parser.add_argument("--validation-split", type=float, default=0.20)
    parser.add_argument("--kmeans-k", type=int, default=16)
    parser.add_argument("--representative-samples", type=int, default=256)
    parser.add_argument("--tflite-eval-samples", type=int, default=0)
    parser.add_argument("--test-digit-index", type=int, default=0)
    return parser.parse_args()


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (
        x_train.astype("float32")[..., None] / 255.0,
        y_train,
        x_test.astype("float32")[..., None] / 255.0,
        y_test,
    )


def split_train_validation(x: np.ndarray, y: np.ndarray, validation_split: float):
    split = float(np.clip(validation_split, 0.0, 0.5))
    n_val = max(1, int(round(len(x) * split))) if split > 0 else 0
    if n_val == 0:
        return x, y, x[:0], y[:0]
    return x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def make_model(conv1: int = 16, conv2: int = 32, dense1: int = 32, dense2: int = 16) -> tf.keras.Model:
    return compile_model(tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1), name="image"),
        tf.keras.layers.Conv2D(conv1, 3, activation="relu", name="conv1"),
        tf.keras.layers.MaxPooling2D(name="pool1"),
        tf.keras.layers.Conv2D(conv2, 3, activation="relu", name="conv2"),
        tf.keras.layers.MaxPooling2D(name="pool2"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(dense1, activation="relu", name="hidden1"),
        tf.keras.layers.Dense(dense2, activation="relu", name="hidden2"),
        tf.keras.layers.Dense(10, activation="softmax", name="output"),
    ]))


def make_student_model() -> tf.keras.Model:
    ## TODO
    ## create a smaller student architecture
    return make_model(8, 16, 16, 8)


def clone_model(model: tf.keras.Model) -> tf.keras.Model:
    clone = tf.keras.models.clone_model(model)
    clone(tf.zeros((1, 28, 28, 1)))
    clone.set_weights([w.copy() for w in model.get_weights()])
    return compile_model(clone)


def keep_count(total: int, prune_ratio: float) -> int:
    return max(1, min(total, int(round(total * (1.0 - float(np.clip(prune_ratio, 0.0, 0.95)))))))


def prune_synapses(model: tf.keras.Model, prune_ratio: float) -> tf.keras.Model:
    pruned = clone_model(model)
    kernels = [np.abs(w[0]).ravel() for w in (layer.get_weights() for layer in pruned.layers) if w]
    threshold = np.percentile(np.concatenate(kernels), prune_ratio * 100.0)
    for layer in pruned.layers:
        weights = layer.get_weights()
        if weights:
            weights[0][np.abs(weights[0]) <= threshold] = 0.0
            layer.set_weights(weights)
    return pruned


def prune_neurons(model: tf.keras.Model, prune_ratio: float) -> tf.keras.Model:
    conv1, conv2 = model.get_layer("conv1"), model.get_layer("conv2")
    h1, h2, out = model.get_layer("hidden1"), model.get_layer("hidden2"), model.get_layer("output")
    h1_k, h1_b = h1.get_weights()
    h2_k, h2_b = h2.get_weights()
    out_k, out_b = out.get_weights()

    keep = np.sort(np.argsort(np.sum(np.abs(h1_k), axis=0) + np.sum(np.abs(h2_k), axis=1))[-keep_count(h1_k.shape[1], prune_ratio):])
    h1_k, h1_b, h2_k = h1_k[:, keep], h1_b[keep], h2_k[keep, :]

    keep = np.sort(np.argsort(np.sum(np.abs(h2_k), axis=0) + np.sum(np.abs(out_k), axis=1))[-keep_count(h2_k.shape[1], prune_ratio):])
    h2_k, h2_b, out_k = h2_k[:, keep], h2_b[keep], out_k[keep, :]

    pruned = make_model(conv1.filters, conv2.filters, h1_k.shape[1], h2_k.shape[1])
    pruned.get_layer("conv1").set_weights(conv1.get_weights())
    pruned.get_layer("conv2").set_weights(conv2.get_weights())
    pruned.get_layer("hidden1").set_weights([h1_k, h1_b])
    pruned.get_layer("hidden2").set_weights([h2_k, h2_b])
    pruned.get_layer("output").set_weights([out_k, out_b])
    return pruned


def rows_for_channels(channels: np.ndarray, old_channels: int, side: int = 5) -> np.ndarray:
    return np.array([(y * side + x) * old_channels + int(c) for y in range(side) for x in range(side) for c in channels], dtype=np.int64)


def prune_channels(model: tf.keras.Model, prune_ratio: float) -> tf.keras.Model:
    conv1, conv2 = model.get_layer("conv1"), model.get_layer("conv2")
    h1, h2, out = model.get_layer("hidden1"), model.get_layer("hidden2"), model.get_layer("output")
    c1_k, c1_b = conv1.get_weights()
    c2_k, c2_b = conv2.get_weights()
    h1_k, h1_b = h1.get_weights()
    old_c2 = c2_k.shape[3]

    keep = np.sort(np.argsort(np.sum(np.abs(c1_k), axis=(0, 1, 2)) + np.sum(np.abs(c2_k), axis=(0, 1, 3)))[-keep_count(c1_k.shape[3], prune_ratio):])
    c1_k, c1_b, c2_k = c1_k[:, :, :, keep], c1_b[keep], c2_k[:, :, keep, :]

    conv_scores = np.sum(np.abs(c2_k), axis=(0, 1, 2))
    hidden_scores = np.sum(np.abs(h1_k.reshape(5, 5, old_c2, h1_k.shape[1])), axis=(0, 1, 3))
    keep = np.sort(np.argsort(conv_scores + hidden_scores)[-keep_count(old_c2, prune_ratio):])
    c2_k, c2_b = c2_k[:, :, :, keep], c2_b[keep]
    h1_k = h1_k[rows_for_channels(keep, old_c2), :]

    pruned = make_model(c1_k.shape[3], c2_k.shape[3], h1.units, h2.units)
    pruned.get_layer("conv1").set_weights([c1_k, c1_b])
    pruned.get_layer("conv2").set_weights([c2_k, c2_b])
    pruned.get_layer("hidden1").set_weights([h1_k, h1_b])
    pruned.get_layer("hidden2").set_weights(h2.get_weights())
    pruned.get_layer("output").set_weights(out.get_weights())
    return pruned


def finetune(model: tf.keras.Model, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> None:
    if epochs > 0:
        model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True)


def soften(probabilities: tf.Tensor, temperature: float) -> tf.Tensor:
    ## TODO
    ## enable softening of softmax with a temperature
    return tf.nn.softmax(tf.nn.log(tf.clip_by_value(probabilities, 1e-8, 1.0)) / temperature, axis=-1)


def distill_student(teacher: tf.keras.Model, student: tf.keras.Model, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> None:
   ## TODO
   ## perform knowledge destillation from a teacher to a student model
    if epochs > 0:
        return
   
    was_trainable = teacher.trainable
    teacher.trainable = False

    optimizer = tf.keras.optimizers.Adam()

    hard_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    soft_loss = tf.keras.losses.KLDivergence()

    dataset = (
       tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=42, reshuffle_each_iteration=True).batch(batch_size)
    )

    distill_temperature = 4
    distill_alpha = 0.5

    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        for batch_x, batch_y in dataset:

            teacher_soft = soften(teacher(batch_x, training=False), distill_temperature)
            with tf.GradientTape() as tape:
                student_probs = student(batch_x, training=True)
                student_soft = soften(student_probs, distill_temperature)
                hard_label_component = hard_loss(batch_y, student_probs)
                distillation_component = soft_loss(teacher_soft, soften(student_probs, distill_temperature))
                loss = distill_alpha * hard_label_component + (1 - distill_alpha) * distillation_component * (distill_temperature ** 2)

            gradient = tape.gradient(loss, student.trainable_variables)
            optimizer.apply_gradients(zip(gradient, student.trainable_variables))

            losses.append(float(loss.numpy))

        print(
            f"Distill epoch {epoch}/{epochs}", 
            f"loss={float(np.mean(losses)):.4f}"
            )


def distill_then_finetune(teacher: tf.keras.Model, student: tf.keras.Model, x: np.ndarray, y: np.ndarray, finetune_epochs: int, batch_size: int) -> None:
    distill_student(teacher, student, x, y, DISTILL_EPOCHS, batch_size)
    finetune(student, x, y, finetune_epochs, batch_size)


def make_kmeans_model(model: tf.keras.Model, k: int, clustering: Any, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> tf.keras.Model:
    clustered = clustering.cluster_weights(
        clone_model(model),
        number_of_clusters=max(1, int(k)),
        cluster_centroids_init=clustering.CentroidInitialization.LINEAR,
    )
    clustered(tf.zeros((1, 28, 28, 1)))
    compile_model(clustered)
    finetune(clustered, x, y, epochs, batch_size)
    stripped = clustering.strip_clustering(clustered)
    return compile_model(stripped)


def representative_dataset(x: np.ndarray, sample_count: int):
    limit = max(1, min(int(sample_count), len(x)))
    def generator():
        for i in range(limit):
            yield [x[i:i + 1].astype(np.float32)]
    return generator


def export_tflite(model: tf.keras.Model, artifacts_dir: Path, name: str, mode: str, representative_data: np.ndarray, representative_samples: int) -> Path:
    keras_dir = artifacts_dir / "keras"
    saved_dir = artifacts_dir / "saved_models" / name
    keras_dir.mkdir(parents=True, exist_ok=True)
    saved_dir.parent.mkdir(parents=True, exist_ok=True)
    if saved_dir.exists():
        shutil.rmtree(saved_dir)
    model.save(str(keras_dir / f"{name}.keras"))
    tf.saved_model.save(model, str(saved_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
    if mode == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset(representative_data, representative_samples)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    path = artifacts_dir / f"{name}.tflite"
    path.write_bytes(converter.convert())
    return path


def quantize_for_tflite(values: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.floating):
        return values.astype(dtype)
    scale, zero_point = detail["quantization"]
    scale = float(scale) if float(scale) else 1.0
    info = np.iinfo(dtype)
    return np.clip(np.round(values / scale + int(zero_point)), info.min, info.max).astype(dtype)


def dequantize_from_tflite(values: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    if not np.issubdtype(values.dtype, np.integer):
        return values.astype(np.float32)
    scale, zero_point = detail["quantization"]
    scale = float(scale) if float(scale) else 1.0
    return (values.astype(np.float32) - int(zero_point)) * scale


def evaluate_tflite(path: Path, x: np.ndarray, y: np.ndarray, sample_limit: int) -> float:
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    limit = len(x) if sample_limit <= 0 else min(sample_limit, len(x))
    correct = 0
    for i in range(limit):
        sample = x[i:i + 1].astype(np.float32)
        interpreter.set_tensor(input_detail["index"], quantize_for_tflite(sample, input_detail))
        interpreter.invoke()
        output = dequantize_from_tflite(interpreter.get_tensor(output_detail["index"]), output_detail)[0]
        correct += int(int(np.argmax(output)) == int(y[i]))
    return float(correct / max(1, limit))


def parameter_count(model: tf.keras.Model) -> int:
    total = 0
    for layer in model.layers:
        for attr in ("kernel", "bias"):
            variable = getattr(layer, attr, None)
            if variable is not None:
                total += int(variable.numpy().size)
    return total


def evaluate_keras(model: tf.keras.Model, x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    val_acc = float(model.evaluate(x_val, y_val, verbose=0)[1]) if len(x_val) else float("nan")
    test_acc = float(model.evaluate(x_test, y_test, verbose=0)[1])
    return val_acc, test_acc


def save_test_digit_bmp(path: Path, image_28x28: np.ndarray, scale: int = 10) -> None:
    image = np.repeat(np.repeat(image_28x28, scale, axis=0), scale, axis=1)
    pixels = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    height, width = pixels.shape
    row_stride = (width * 3 + 3) & ~3
    pixel_data_size = row_stride * height
    with path.open("wb") as output:
        output.write(b"BM")
        output.write(struct.pack("<IHHI", 54 + pixel_data_size, 0, 0, 54))
        output.write(struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, pixel_data_size, 0, 0, 0, 0))
        padding = b"\x00" * (row_stride - width * 3)
        for y in range(height - 1, -1, -1):
            row = bytearray()
            for value in pixels[y]:
                row.extend([int(value), int(value), int(value)])
            output.write(row)
            output.write(padding)


def append_model(models: list[dict[str, Any]], name: str, method: str, model: tf.keras.Model, mode: str = "fp32") -> None:
    models.append({"name": name, "method": method, "model": model, "mode": mode})


def main() -> int:
    args = parse_args()
    clustering = require_tfmot()
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    x_train_full, y_train_full, x_test, y_test = load_data()
    x_train, y_train, x_val, y_val = split_train_validation(x_train_full, y_train_full, args.validation_split)

    test_index = args.test_digit_index % len(x_test)
    save_test_digit_bmp(artifacts_dir / "test_digit.bmp", x_test[test_index, :, :, 0])
    (artifacts_dir / "test_digit_label.txt").write_text(f"{int(y_test[test_index])}\n", encoding="utf-8")

    baseline = make_model(args.conv_filters, args.conv_filters * 2, args.dense_units, max(1, args.dense_units // 2))
    baseline.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val) if len(x_val) else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        shuffle=True,
    )

    models: list[dict[str, Any]] = []
    append_model(models, "baseline", "normal FP32 training", baseline)

    synapse = prune_synapses(baseline, args.synapse_prune_ratio)
    finetune(synapse, x_train, y_train, args.finetune_epochs, args.batch_size)
    synapse = prune_synapses(synapse, args.synapse_prune_ratio)
    append_model(models, "synapse_pruned", "unstructured small-weight pruning", synapse)

    neuron = prune_neurons(baseline, args.neuron_prune_ratio)
    finetune(neuron, x_train, y_train, args.finetune_epochs, args.batch_size)
    append_model(models, "neuron_pruned", "structured hidden-neuron pruning", neuron)

    channel = prune_channels(baseline, args.channel_prune_ratio)
    finetune(channel, x_train, y_train, args.finetune_epochs, args.batch_size)
    append_model(models, "channel_pruned", "structured convolution-channel pruning", channel)

    student = make_student_model()
    ## TODO
    ## perform knowledge distillation to a smaller student model
    distill_student(baseline, student, x_train, y_train, DISTILL_EPOCHS, args.batch_size)
    append_model(models, "student_kd", "true small student knowledge distillation", student)    

    append_model(models, "baseline_fp16", "FP16 TFLite export", baseline, "fp16")
    append_model(models, "baseline_int8", "INT8 TFLite export", baseline, "int8")

    kmeans = make_kmeans_model(baseline, args.kmeans_k, clustering, x_train, y_train, args.cluster_finetune_epochs, args.batch_size)
    append_model(models, f"kmeans_k{int(args.kmeans_k)}", "standalone TFMOT K-Means weight clustering example", kmeans)

    ## TODO
    ## create a neuron and channel pruned version of the student and 
    ## then perform knowledge distillation again to keep performance high
    ## afterwards finetune to the actual labels and quantize to int8 for a as small as possible model
    final_student = prune_channels(student, args.channel_prune_ratio)
    final_student = prune_neurons(final_student, args.neuron_prune_ratio)
    distill_then_finetune(baseline, final_student, x_train, y_train, args.finetune_epochs, args.batch_size)
    append_model(models, "student_channel_neuron_pruned_kd", "true studen KD + channel and neuron pruning + INT8 TFLite export", final_student, "int8")

    rows: list[dict[str, Any]] = []
    for item in models:
        model = item["model"]
        mode = str(item["mode"])
        val_acc, test_acc = evaluate_keras(model, x_val, y_val, x_test, y_test)
        tflite_path = export_tflite(model, artifacts_dir, str(item["name"]), mode, x_train, args.representative_samples)
        tflite_acc = evaluate_tflite(tflite_path, x_test, y_test, args.tflite_eval_samples)
        rows.append({
            "tflite_file": tflite_path.name,
            "method": str(item["method"]),
            "tflite_mode": mode,
            "validation_accuracy": f"{val_acc:.3f}",
            "test_accuracy": f"{test_acc:.3f}",
            "tflite_test_accuracy": f"{tflite_acc:.3f}",
            "tflite_file_size_kilobytes": int(tflite_path.stat().st_size / 1024),
            "parameters_total": parameter_count(model),
        })
        print(f"Exported {tflite_path} keras_acc={test_acc:.4f} tflite_acc={tflite_acc:.4f}")

    shutil.copy2(artifacts_dir / "baseline.tflite", artifacts_dir / "model.tflite")
    metrics_path = artifacts_dir / "model_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved benchmark BMP: {artifacts_dir / 'test_digit.bmp'}")
    print(f"Saved test digit label: {int(y_test[test_index])}")
    print(f"Saved model metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
