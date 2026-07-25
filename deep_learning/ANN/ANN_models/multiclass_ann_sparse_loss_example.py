import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


tf.get_logger().setLevel("ERROR")
device_name = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print(f"TensorFlow {tf.__version__} is running on {device_name}.", flush=True)


# ============================================================
# Multiclass ANN: 24 And 12 Hidden Neurons + Adam + Sparse Loss
# ============================================================
#
# This network predicts one of three iris species. Its architecture is:
# 4 inputs -> 24 hidden neurons -> 12 hidden neurons -> 3 output neurons
#
# There are three output neurons because there are three possible classes.
# Softmax makes their three probabilities add up to 1.


keras.utils.set_random_seed(42)

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.20,
    stratify=data.target,
    random_state=42,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = keras.Sequential(
    [
        keras.layers.Input(shape=(X_train.shape[1],)),

        # More neurons give a layer more capacity to learn patterns. However,
        # unnecessarily large layers can train slowly and overfit small data.
        keras.layers.Dense(units=24, activation="relu"),
        keras.layers.Dense(units=12, activation="relu"),

        # units=3 creates one score per iris class. Softmax changes the scores
        # into a probability distribution across all three classes.
        keras.layers.Dense(units=3, activation="softmax"),
    ],
    name="iris_sparse_adam_ann",
)


optimizer = keras.optimizers.Adam(
    # Adam adapts the update size separately for different model weights.
    # 0.001 is a common starting learning rate for Adam.
    learning_rate=0.001,
)

model.compile(
    optimizer=optimizer,
    # SparseCategoricalCrossentropy expects class labels stored as integers,
    # such as 0, 1, and 2. It does not require one-hot-encoded targets.
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # SparseCategoricalAccuracy also compares integer labels with softmax output.
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=80,
    batch_size=16,
    validation_split=0.20,
    # Show live epoch progress while the model trains.
    verbose=1,
)


# Each row contains three probabilities. argmax returns the index of the
# largest probability, which is the network's selected class.
y_probability = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_probability, axis=1)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nInteger target example:", y_train[:8])
print("Final training accuracy:", round(history.history["accuracy"][-1], 3))
print("Final validation accuracy:", round(history.history["val_accuracy"][-1], 3))
print("Test sparse categorical cross-entropy:", round(test_loss, 3))
print("Test accuracy:", round(test_accuracy, 3))
print("\nFirst three softmax probability rows:")
print(np.round(y_probability[:3], 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
