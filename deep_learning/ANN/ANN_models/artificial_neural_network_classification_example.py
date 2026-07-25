import os

# Request less TensorFlow C++ output. TensorFlow 2.21 may still print its two
# oneDNN notices before logging initializes; oneDNN remains enabled for speed.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Native Windows supports TensorFlow CPU training. TensorFlow 2.11+ does not
# support NVIDIA GPU acceleration on native Windows, but that does not prevent
# models from training on an Intel or AMD CPU.
tf.get_logger().setLevel("ERROR")
device_name = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print(f"TensorFlow {tf.__version__} is running on {device_name}.", flush=True)


# ============================================================
# Artificial Neural Network (ANN) Classification With Keras
# ============================================================
#
# This beginner example demonstrates how to:
# - load classification data from sklearn
# - prepare the features for a neural network
# - create hidden layers and choose their numbers of neurons
# - train the network with TensorFlow/Keras
# - predict probabilities and class labels
# - evaluate the predictions with classification metrics
#
# An ANN learns by passing feature values through connected layers of neurons.
# During training, it changes the weight of each connection to reduce its error.


# Set random seeds so weight initialization and training are more reproducible.
# Exact results can still differ slightly across hardware and TensorFlow versions.
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 1. Load and split the data
# ============================================================

# The breast-cancer dataset is a binary classification dataset built into
# sklearn. Each row describes a tumor using 30 numerical measurements.
data = load_breast_cancer()
X = data.data
y = data.target

# y contains two classes:
# 0 = malignant
# 1 = benign
print("Feature matrix shape:", X.shape)
print("Target class names:", list(data.target_names))


# First reserve 20% of all rows for the final test set.
# The test set must not be used while fitting the scaler or training the ANN.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    # test_size=0.20 sends 20% of the rows to the test set.
    test_size=0.20,
    # random_state makes the same split each time the script runs.
    random_state=42,
    # stratify=y keeps similar class proportions in both sets.
    stratify=y,
)


# ============================================================
# 2. Scale the numerical features
# ============================================================

# StandardScaler changes each feature to have approximately:
# - mean = 0
# - standard deviation = 1
#
# Neural networks generally train more reliably when input features are on
# comparable scales. fit_transform learns scaling values from training data.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# transform reuses the training means and standard deviations. We must not call
# fit_transform on the test set because that would leak test information.
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 3. Build the ANN and define its layers
# ============================================================

# Sequential means that data flows through each layer in the listed order.
model = keras.Sequential(
    [
        # Input describes the shape of one sample. The dataset has 30 features,
        # so input_shape is (30,). The comma denotes a one-dimensional tuple.
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),

        # First hidden layer:
        # - units=16 creates 16 neurons.
        # - activation="relu" returns max(0, value). ReLU lets the network
        #   learn nonlinear relationships and is common in hidden layers.
        keras.layers.Dense(units=16, activation="relu"),

        # Second hidden layer:
        # - units=8 creates a smaller layer of 8 neurons.
        # - this layer learns combinations of patterns found by the first layer.
        keras.layers.Dense(units=8, activation="relu"),

        # Output layer:
        # - units=1 is used because this is binary classification.
        # - sigmoid converts the output into a probability between 0 and 1.
        keras.layers.Dense(units=1, activation="sigmoid"),
    ],
    # name gives the model a readable label in model.summary().
    name="breast_cancer_ann",
)


# Display every layer, its output shape, and its number of trainable parameters.
print("\nANN architecture:")
model.summary()


# ============================================================
# 4. Configure how the ANN will learn
# ============================================================

model.compile(
    # optimizer="adam" selects the Adam optimizer. It adjusts the weights using
    # gradients and automatically adapts the learning rate for each weight.
    optimizer="adam",

    # binary_crossentropy measures error between the true binary labels and the
    # predicted probabilities. It is the standard loss for sigmoid binary output.
    loss="binary_crossentropy",

    # metrics=["accuracy"] asks Keras to report the fraction classified correctly.
    metrics=["accuracy"],
)


# ============================================================
# 5. Train the ANN
# ============================================================

history = model.fit(
    X_train_scaled,
    y_train,
    # epochs=30 means the network sees the complete training set 30 times.
    epochs=30,
    # batch_size=32 updates weights after each group of 32 training samples.
    batch_size=32,
    # validation_split=0.20 uses 20% of the training rows to monitor how well
    # the network generalizes during training. It does not use the test set.
    validation_split=0.20,
    # verbose=1 displays live epoch progress and metric values.
    verbose=1,
)


# history.history stores one value per epoch for every tracked measurement.
print("\nFinal training accuracy:", round(history.history["accuracy"][-1], 3))
print("Final validation accuracy:", round(history.history["val_accuracy"][-1], 3))
print("Final training loss:", round(history.history["loss"][-1], 3))
print("Final validation loss:", round(history.history["val_loss"][-1], 3))


# ============================================================
# 6. Evaluate and predict
# ============================================================

# evaluate computes the configured loss and accuracy on unseen test data.
# verbose=0 prevents Keras from printing a progress bar.
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)


# predict returns sigmoid probabilities with shape (number_of_rows, 1).
# ravel changes that two-dimensional result into a one-dimensional array.
y_probability = model.predict(X_test_scaled, verbose=0).ravel()

# Convert probabilities into class labels using a threshold of 0.5:
# probability >= 0.5 becomes class 1, otherwise it becomes class 0.
y_pred = (y_probability >= 0.5).astype(int)


print("\nTest loss:", round(test_loss, 3))
print("Keras test accuracy:", round(test_accuracy, 3))
print("sklearn test accuracy:", round(accuracy_score(y_test, y_pred), 3))

print("\nFirst five predicted probabilities:")
print(np.round(y_probability[:5], 3))

print("\nFirst five predicted classes:")
print(y_pred[:5])

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print(
    "Interpretation: the hidden layers learn patterns from the scaled tumor"
    " measurements. The final sigmoid neuron estimates the probability of"
    " class 1, and the 0.5 threshold converts that probability into a class."
)
