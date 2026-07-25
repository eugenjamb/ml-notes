import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


tf.get_logger().setLevel("ERROR")
device_name = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print(f"TensorFlow {tf.__version__} is running on {device_name}.", flush=True)


# ============================================================
# Multiclass ANN: 48 And 24 Hidden Neurons + RMSprop + One-Hot Loss
# ============================================================
#
# This example uses a wider network than the iris example:
# 13 inputs -> 48 hidden neurons -> 24 hidden neurons -> 3 output neurons
#
# It also converts target classes to one-hot vectors and therefore uses
# CategoricalCrossentropy instead of SparseCategoricalCrossentropy.


keras.utils.set_random_seed(42)

data = load_wine()
X_train, X_test, y_train_integer, y_test_integer = train_test_split(
    data.data,
    data.target,
    test_size=0.20,
    stratify=data.target,
    random_state=42,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# to_categorical changes integer labels into one-hot vectors:
# class 0 -> [1, 0, 0]
# class 1 -> [0, 1, 0]
# class 2 -> [0, 0, 1]
# num_classes=3 sets the vector width to the number of possible classes.
y_train = keras.utils.to_categorical(y_train_integer, num_classes=3)
y_test = keras.utils.to_categorical(y_test_integer, num_classes=3)


model = keras.Sequential(
    [
        keras.layers.Input(shape=(X_train.shape[1],)),

        # units=48 creates a relatively wide first hidden layer. It can learn
        # many combinations of the 13 chemical measurements.
        keras.layers.Dense(units=48, activation="relu"),

        # units=24 gradually compresses the representation before prediction.
        keras.layers.Dense(units=24, activation="relu"),

        # Three softmax neurons match the three columns in each one-hot target.
        keras.layers.Dense(units=3, activation="softmax"),
    ],
    name="wine_one_hot_rmsprop_ann",
)


optimizer = keras.optimizers.RMSprop(
    # RMSprop adapts weight updates using a moving average of squared gradients.
    learning_rate=0.001,
    # rho controls how much previous squared gradients influence that average.
    rho=0.9,
    # epsilon prevents division by zero during optimizer calculations.
    epsilon=1e-7,
)

model.compile(
    optimizer=optimizer,
    # CategoricalCrossentropy expects one-hot targets. It compares all target
    # columns with the corresponding softmax probabilities.
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.20,
    # Show live epoch progress while the model trains.
    verbose=1,
)


y_probability = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_probability, axis=1)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nFirst five one-hot training targets:")
print(y_train[:5].astype(int))
print("\nFinal training accuracy:", round(history.history["accuracy"][-1], 3))
print("Final validation accuracy:", round(history.history["val_accuracy"][-1], 3))
print("Test categorical cross-entropy:", round(test_loss, 3))
print("Test accuracy:", round(test_accuracy, 3))
print("\nFirst three softmax probability rows:")
print(np.round(y_probability[:3], 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test_integer, y_pred))
print("\nClassification report:")
print(classification_report(y_test_integer, y_pred, target_names=data.target_names))
