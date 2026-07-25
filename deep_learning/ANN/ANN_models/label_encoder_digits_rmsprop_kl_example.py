import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import keras
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


tf.get_logger().setLevel("ERROR")
device_name = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print(f"TensorFlow {tf.__version__} is running on {device_name}.", flush=True)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LabelEncoder Multiclass ANN: ELU + RMSprop + KL Divergence
# ============================================================
#
# This example predicts ten string labels: digit_0 through digit_9.
# It demonstrates KLDivergence as an alternative loss for one-hot probability
# distributions. For one-hot targets, it behaves similarly to categorical
# cross-entropy because the target distribution contains one class with value 1.
#
# Architecture: 64 inputs -> 64 ELU neurons -> 32 ELU neurons -> 10 outputs


keras.utils.set_random_seed(42)

data = load_digits()

# Each 8 x 8 image is already flattened into 64 pixel features.
X = data.data.astype("float32")
y_text = np.array([f"digit_{digit}" for digit in data.target])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_text)
number_of_classes = len(label_encoder.classes_)
y_one_hot = keras.utils.to_categorical(y_encoded, num_classes=number_of_classes)

print("Number of encoded classes:", number_of_classes)
print("LabelEncoder classes:", label_encoder.classes_.tolist())


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

# Keep unscaled copies so the unseen 8 x 8 images can be plotted later.
X_test_images = X_test.copy()

# Digits pixel values range from 0 to 16. Dividing by 16 maps them to 0..1,
# which helps neural-network optimization without fitting a separate scaler.
X_train = X_train / 16.0
X_test = X_test / 16.0


model = keras.Sequential(
    [
        keras.layers.Input(shape=(X_train.shape[1],)),
        # ELU behaves like a linear function for positive inputs and has a
        # smooth negative region. It can keep gradients active below zero.
        keras.layers.Dense(units=64, activation="elu"),
        keras.layers.Dense(units=32, activation="elu"),
        keras.layers.Dense(units=number_of_classes, activation="softmax"),
    ],
    name="digits_label_encoder_rmsprop_ann",
)


optimizer = keras.optimizers.RMSprop(
    # RMSprop scales updates using a moving average of squared gradients.
    learning_rate=0.001,
    # rho controls how strongly older squared gradients affect that average.
    rho=0.9,
)

model.compile(
    optimizer=optimizer,
    # KLDivergence compares the one-hot target distribution with the softmax
    # probability distribution predicted by the ANN.
    loss=keras.losses.KLDivergence(),
    metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
)

print("Training the Digits ANN for up to 20 epochs...", flush=True)
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
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
prediction_confidence = np.max(y_probability, axis=1)
incorrect_mask = y_pred_encoded != y_test_encoded

print("Test KL-divergence loss:", round(test_loss, 3))
print("Keras test accuracy:", round(test_accuracy, 3))
print("sklearn accuracy:", round(sklearn_accuracy, 3))
print("Balanced accuracy:", round(balanced_accuracy, 3))
print("Incorrect unseen predictions:", int(np.sum(incorrect_mask)))
print("Mean prediction confidence:", round(float(np.mean(prediction_confidence)), 3))

print("\nPredictions for the first ten unseen digit images:")
for row_number in range(10):
    print(
        f"actual={y_test_text[row_number]:7s} "
        f"predicted={y_pred_text[row_number]:7s} "
        f"confidence={prediction_confidence[row_number]:.3f}"
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


# Plot 1: learning curves reveal convergence and possible overfitting.
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs = range(1, len(history.history["loss"]) + 1)
axes[0].plot(epochs, history.history["accuracy"], label="Training")
axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
axes[0].set(title="Digits accuracy by epoch", xlabel="Epoch", ylabel="Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(epochs, history.history["loss"], label="Training")
axes[1].plot(epochs, history.history["val_loss"], label="Validation")
axes[1].set(title="Digits loss by epoch", xlabel="Epoch", ylabel="Loss")
axes[1].legend()
axes[1].grid(alpha=0.25)
figure.tight_layout()
learning_curve_path = OUTPUT_DIR / "digits_label_encoder_learning_curves.png"
figure.savefig(learning_curve_path, dpi=150)


# Plot 2: a ten-class confusion matrix summarizes all unseen predictions.
figure, axis = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay(
    confusion_matrix=test_confusion_matrix,
    display_labels=label_encoder.classes_,
).plot(ax=axis, cmap="Oranges", colorbar=False, xticks_rotation=45)
axis.set_title("Digits unseen test confusion matrix")
figure.tight_layout()
confusion_matrix_path = OUTPUT_DIR / "digits_label_encoder_confusion_matrix.png"
figure.savefig(confusion_matrix_path, dpi=150)


# Plot 3: inspect actual unseen images. Green titles are correct predictions;
# red titles are mistakes, making model failures easy to investigate.
figure, axes = plt.subplots(2, 5, figsize=(11, 5))
for row_number, axis in enumerate(axes.flat):
    axis.imshow(X_test_images[row_number].reshape(8, 8), cmap="gray_r")
    is_correct = y_pred_encoded[row_number] == y_test_encoded[row_number]
    title_color = "green" if is_correct else "red"
    axis.set_title(
        f"actual: {y_test_text[row_number]}\npred: {y_pred_text[row_number]}",
        color=title_color,
        fontsize=9,
    )
    axis.axis("off")

figure.suptitle("Predictions on ten unseen digit images")
figure.tight_layout()
image_grid_path = OUTPUT_DIR / "digits_label_encoder_unseen_predictions.png"
figure.savefig(image_grid_path, dpi=150)

print("\nSaved graphs:")
print(learning_curve_path)
print(confusion_matrix_path)
print(image_grid_path)
plt.show()
