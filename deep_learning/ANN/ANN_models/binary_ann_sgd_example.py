import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


tf.get_logger().setLevel("ERROR")
device_name = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print(f"TensorFlow {tf.__version__} is running on {device_name}.", flush=True)


# ============================================================
# Binary ANN: 12 And 6 Hidden Neurons + SGD + Binary Loss
# ============================================================
#
# This network predicts one of two classes. Its architecture is:
# 30 inputs -> 12 hidden neurons -> 6 hidden neurons -> 1 output neuron
#
# The output uses one sigmoid neuron because binary classification needs one
# probability: the probability that a sample belongs to class 1.


# Sets Python, NumPy, and TensorFlow random seeds through one Keras helper.
keras.utils.set_random_seed(42)

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    # test_size=0.20 reserves 20% for final evaluation.
    test_size=0.20,
    # stratify preserves the malignant/benign class ratio.
    stratify=data.target,
    random_state=42,
)


# Neural networks learn more easily when numeric features have similar scales.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = keras.Sequential(
    [
        keras.layers.Input(shape=(X_train.shape[1],)),

        # A neuron is a small calculation with learned weights and a bias.
        # units=12 means this layer learns 12 different feature combinations.
        keras.layers.Dense(units=12, activation="relu"),

        # A smaller second layer compresses the 12 learned patterns into 6
        # higher-level patterns before the final prediction.
        keras.layers.Dense(units=6, activation="relu"),

        # One sigmoid output produces a value from 0 to 1.
        keras.layers.Dense(units=1, activation="sigmoid"),
    ],
    name="binary_sgd_ann",
)


# SGD means stochastic gradient descent. It moves model weights in the
# direction that reduces loss.
optimizer = keras.optimizers.SGD(
    # learning_rate controls the size of each weight update. Too large may
    # overshoot a solution; too small can make learning very slow.
    learning_rate=0.01,
    # momentum remembers part of the previous update, which can reduce
    # oscillation and help SGD move through shallow regions more quickly.
    momentum=0.9,
)

model.compile(
    optimizer=optimizer,
    # BinaryCrossentropy compares binary labels (0 or 1) with sigmoid
    # probabilities. It matches an output layer with one sigmoid neuron.
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    # One epoch is one complete pass through the training rows.
    epochs=40,
    # Weights are updated after each batch of 32 rows.
    batch_size=32,
    # Keras monitors 20% of the training data without learning from those rows.
    validation_split=0.20,
    # Show the epoch counter and metrics so training progress is visible.
    verbose=1,
)


# predict returns probabilities. The 0.5 threshold converts them to 0 or 1.
y_probability = model.predict(X_test, verbose=0).ravel()
y_pred = (y_probability >= 0.5).astype(int)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nFinal training accuracy:", round(history.history["accuracy"][-1], 3))
print("Final validation accuracy:", round(history.history["val_accuracy"][-1], 3))
print("Test binary cross-entropy:", round(test_loss, 3))
print("Test accuracy:", round(test_accuracy, 3))
print("\nFirst five probabilities:", np.round(y_probability[:5], 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
