import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import keras
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


tf.get_logger().setLevel("ERROR")
device_name = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print(f"TensorFlow {tf.__version__} is running on {device_name}.", flush=True)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LabelEncoder Multiclass ANN: tanh + SGD + Categorical Loss
# ============================================================
#
# This example demonstrates two target transformations:
# 1. LabelEncoder changes string labels into integer class IDs.
# 2. to_categorical changes each integer into a one-hot vector.
#
# Architecture: 13 inputs -> 32 tanh neurons -> 16 tanh neurons -> 3 outputs


keras.utils.set_random_seed(42)

data = load_wine()
X = data.data
y_text = np.array([data.target_names[class_id] for class_id in data.target])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_text)
number_of_classes = len(label_encoder.classes_)

# CategoricalCrossentropy expects one-hot targets rather than integer targets.
# Example with three classes: class 1 becomes [0, 1, 0].
y_one_hot = keras.utils.to_categorical(
    y_encoded,
    num_classes=number_of_classes,
)

print("LabelEncoder classes:", label_encoder.classes_.tolist())
print("First encoded class ID:", y_encoded[0])
print("Its one-hot representation:", y_one_hot[0].astype(int))


# Split both the one-hot labels used for training and integer labels used later
# for readable predictions and sklearn metrics.
(
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_encoded,
    y_test_encoded,
) = train_test_split(
    X,
    y_one_hot,
    y_encoded,
    test_size=0.20,
    stratify=y_encoded,
    random_state=42,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = keras.Sequential(
    [
        keras.layers.Input(shape=(X_train.shape[1],)),
        # tanh returns values from -1 to 1. It is smooth but can saturate for
        # very large inputs, which is one reason feature scaling matters.
        keras.layers.Dense(units=32, activation="tanh"),
        keras.layers.Dense(units=16, activation="tanh"),
        keras.layers.Dense(units=number_of_classes, activation="softmax"),
    ],
    name="wine_label_encoder_sgd_ann",
)


optimizer = keras.optimizers.SGD(
    # SGD uses one fixed base learning rate rather than Adam-style adaptation.
    learning_rate=0.02,
    # Momentum carries part of the previous update into the next update.
    momentum=0.9,
)

model.compile(
    optimizer=optimizer,
    # CategoricalCrossentropy matches the one-hot y_train matrix.
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
)

print("Training the Wine ANN for up to 35 epochs...", flush=True)
history = model.fit(
    X_train,
    y_train,
    epochs=35,
    batch_size=16,
    validation_split=0.20,
    verbose=0,
)


y_probability = model.predict(X_test, verbose=0)
y_pred_encoded = np.argmax(y_probability, axis=1)
y_pred_text = label_encoder.inverse_transform(y_pred_encoded)
y_test_text = label_encoder.inverse_transform(y_test_encoded)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
sklearn_accuracy = accuracy_score(y_test_text, y_pred_text)
balanced_accuracy = balanced_accuracy_score(y_test_text, y_pred_text)

print("Test categorical loss:", round(test_loss, 3))
print("Keras test accuracy:", round(test_accuracy, 3))
print("sklearn accuracy:", round(sklearn_accuracy, 3))
print("Balanced accuracy:", round(balanced_accuracy, 3))

print("\nPredictions for the first five unseen test rows:")
for row_number in range(5):
    confidence = y_probability[row_number, y_pred_encoded[row_number]]
    print(
        f"actual={y_test_text[row_number]:7s} "
        f"predicted={y_pred_text[row_number]:7s} "
        f"confidence={confidence:.3f}"
    )

print("\nConfusion matrix:")
test_confusion_matrix = confusion_matrix(
    y_test_text,
    y_pred_text,
    labels=label_encoder.classes_,
)
print(test_confusion_matrix)
print("\nClassification report:")
print(
    classification_report(
        y_test_text,
        y_pred_text,
        labels=label_encoder.classes_,
        digits=3,
    )
)


# Plot 1: show validation accuracy and loss while the network was learning.
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs = range(1, len(history.history["loss"]) + 1)
axes[0].plot(epochs, history.history["accuracy"], label="Training")
axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
axes[0].set(title="Wine accuracy by epoch", xlabel="Epoch", ylabel="Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(epochs, history.history["loss"], label="Training")
axes[1].plot(epochs, history.history["val_loss"], label="Validation")
axes[1].set(title="Wine loss by epoch", xlabel="Epoch", ylabel="Loss")
axes[1].legend()
axes[1].grid(alpha=0.25)
figure.tight_layout()
learning_curve_path = OUTPUT_DIR / "wine_label_encoder_learning_curves.png"
figure.savefig(learning_curve_path, dpi=150)


# Plot 2: compare the test confusion matrix with the softmax probabilities for
# one unseen row. The tallest bar is the class selected by argmax.
figure, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(
    confusion_matrix=test_confusion_matrix,
    display_labels=label_encoder.classes_,
).plot(ax=axes[0], cmap="Greens", colorbar=False)
axes[0].set_title("Wine unseen test confusion matrix")

axes[1].bar(label_encoder.classes_, y_probability[0], color="#d97706")
axes[1].set_ylim(0, 1)
axes[1].set_title(
    f"One unseen row: actual {y_test_text[0]}, predicted {y_pred_text[0]}"
)
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Predicted probability")
figure.tight_layout()
prediction_graph_path = OUTPUT_DIR / "wine_label_encoder_test_diagnostics.png"
figure.savefig(prediction_graph_path, dpi=150)

print("\nSaved graphs:")
print(learning_curve_path)
print(prediction_graph_path)
plt.show()
