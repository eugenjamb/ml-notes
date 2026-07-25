import os
from pathlib import Path

# Request less TensorFlow logging while keeping oneDNN CPU optimizations.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import keras
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
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
# LabelEncoder Multiclass ANN: ReLU + Adam + Sparse Loss
# ============================================================
#
# This example starts with text labels such as "setosa" and uses LabelEncoder
# to convert them into the integers required by this ANN:
#
# setosa -> 0, versicolor -> 1, virginica -> 2
#
# Architecture: 4 inputs -> 24 ReLU neurons -> 12 ReLU neurons -> 3 outputs


keras.utils.set_random_seed(42)

data = load_iris()
X = data.data

# sklearn stores this target numerically, so convert it to species names first.
# This lets the example demonstrate a realistic text-label workflow.
y_text = np.array([data.target_names[class_id] for class_id in data.target])


label_encoder = LabelEncoder()

# fit_transform learns the alphabetically sorted class mapping and transforms
# every text label into an integer from 0 to number_of_classes - 1.
y_encoded = label_encoder.fit_transform(y_text)
number_of_classes = len(label_encoder.classes_)

print("LabelEncoder classes:", label_encoder.classes_.tolist())
print("First five text labels:", y_text[:5])
print("First five encoded labels:", y_encoded[:5])


X_train, X_test, y_train, y_test = train_test_split(
    X,
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
        # ReLU changes negative values to zero and keeps positive values.
        keras.layers.Dense(units=24, activation="relu"),
        keras.layers.Dense(units=12, activation="relu"),
        # One softmax neuron is required for each encoded class.
        keras.layers.Dense(units=number_of_classes, activation="softmax"),
    ],
    name="iris_label_encoder_adam_ann",
)


model.compile(
    # Adam adapts the learning rate separately for different weights.
    optimizer=keras.optimizers.Adam(learning_rate=0.005),

    # SparseCategoricalCrossentropy is correct because LabelEncoder produced
    # one integer per row. We do not need to one-hot encode y_train.
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

print("Training the Iris ANN for up to 40 epochs...", flush=True)
history = model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=16,
    validation_split=0.20,
    verbose=0,
)


y_probability = model.predict(X_test, verbose=0)
y_pred_encoded = np.argmax(y_probability, axis=1)

# inverse_transform converts predicted integers back into readable class names.
y_pred_text = label_encoder.inverse_transform(y_pred_encoded)
y_test_text = label_encoder.inverse_transform(y_test)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
sklearn_accuracy = accuracy_score(y_test_text, y_pred_text)
balanced_accuracy = balanced_accuracy_score(y_test_text, y_pred_text)

print("Test sparse categorical loss:", round(test_loss, 3))
print("Keras test accuracy:", round(test_accuracy, 3))
print("sklearn accuracy:", round(sklearn_accuracy, 3))
print("Balanced accuracy:", round(balanced_accuracy, 3))

# X_test is unseen data: these rows were not used to update model weights.
print("\nPredictions for the first five unseen test rows:")
for row_number in range(5):
    confidence = y_probability[row_number, y_pred_encoded[row_number]]
    print(
        f"actual={y_test_text[row_number]:10s} "
        f"predicted={y_pred_text[row_number]:10s} "
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


# Plot 1: learning curves compare training and validation performance by epoch.
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs = range(1, len(history.history["loss"]) + 1)

axes[0].plot(epochs, history.history["accuracy"], label="Training accuracy")
axes[0].plot(epochs, history.history["val_accuracy"], label="Validation accuracy")
axes[0].set(title="Iris accuracy by epoch", xlabel="Epoch", ylabel="Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(epochs, history.history["loss"], label="Training loss")
axes[1].plot(epochs, history.history["val_loss"], label="Validation loss")
axes[1].set(title="Iris loss by epoch", xlabel="Epoch", ylabel="Loss")
axes[1].legend()
axes[1].grid(alpha=0.25)

figure.tight_layout()
learning_curve_path = OUTPUT_DIR / "iris_label_encoder_learning_curves.png"
figure.savefig(learning_curve_path, dpi=150)


# Plot 2: the confusion matrix shows which unseen classes were confused.
figure, axis = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix=test_confusion_matrix,
    display_labels=label_encoder.classes_,
).plot(ax=axis, cmap="Blues", colorbar=False)
axis.set_title("Iris unseen test predictions")
figure.tight_layout()
confusion_matrix_path = OUTPUT_DIR / "iris_label_encoder_confusion_matrix.png"
figure.savefig(confusion_matrix_path, dpi=150)

print("\nSaved graphs:")
print(learning_curve_path)
print(confusion_matrix_path)
plt.show()
