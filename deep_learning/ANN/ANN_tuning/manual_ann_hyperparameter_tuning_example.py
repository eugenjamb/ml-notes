import math
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
# Manual ANN Hyperparameter Tuning
# ============================================================
#
# Common ANN hyperparameters that developers can tune:
# - number of hidden layers: controls the depth of the network
# - units per hidden layer: controls each layer's learning capacity
# - activation: for example relu, tanh, gelu, or sigmoid
# - optimizer: for example Adam, RMSprop, or SGD
# - learning rate: controls the size of optimizer weight updates
# - batch_size: samples used before one weight update
# - maximum epochs: maximum complete passes through the training data
# - dropout/L1/L2: regularization that can reduce overfitting
#
# We tune batch_size, not "number of batches" directly. Keras calculates:
# batches per epoch = ceil(number of training samples / batch_size)
#
# Loss is normally selected from the task and target format, not searched
# blindly. Binary labels + one sigmoid output require binary cross-entropy.


keras.utils.set_random_seed(42)

data = load_breast_cancer()

# Keep a test set completely outside hyperparameter selection.
X_development, X_test, y_development, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.20,
    stratify=data.target,
    random_state=42,
)

# The validation set compares candidate configurations during tuning.
X_train, X_validation, y_train, y_validation = train_test_split(
    X_development,
    y_development,
    test_size=0.20,
    stratify=y_development,
    random_state=42,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)


def build_model(units_1, units_2, activation, optimizer_name, learning_rate):
    """Create a fresh ANN from one hyperparameter configuration."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(units=units_1, activation=activation),
            keras.layers.Dense(units=units_2, activation=activation),
            # One sigmoid output matches binary labels 0 and 1.
            keras.layers.Dense(units=1, activation="sigmoid"),
        ]
    )

    # Learning rate belongs to the optimizer, so construct the selected
    # optimizer after the configuration supplies its learning rate.
    optimizers = {
        "adam": keras.optimizers.Adam(learning_rate=learning_rate),
        "rmsprop": keras.optimizers.RMSprop(learning_rate=learning_rate),
        "sgd": keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
        ),
    }

    model.compile(
        optimizer=optimizers[optimizer_name],
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# A manual search is useful for a small set of informed experiments. It is not
# a complete Cartesian grid; each dictionary is one intentional experiment.
candidate_configs = [
    {
        "units_1": 16,
        "units_2": 8,
        "activation": "relu",
        "optimizer_name": "adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "max_epochs": 20,
    },
    {
        "units_1": 32,
        "units_2": 16,
        "activation": "relu",
        "optimizer_name": "adam",
        "learning_rate": 0.0005,
        "batch_size": 32,
        "max_epochs": 25,
    },
    {
        "units_1": 24,
        "units_2": 12,
        "activation": "tanh",
        "optimizer_name": "rmsprop",
        "learning_rate": 0.001,
        "batch_size": 64,
        "max_epochs": 25,
    },
]


best_config = None
best_validation_accuracy = -np.inf
trial_results = []

for trial_number, config in enumerate(candidate_configs, start=1):
    print(
        f"\nStarting manual trial {trial_number}/{len(candidate_configs)}:"
        f" {config}",
        flush=True,
    )

    # Clear models from the previous trial before creating a new one.
    keras.backend.clear_session()
    keras.utils.set_random_seed(42 + trial_number)

    model_parameters = {
        key: config[key]
        for key in [
            "units_1",
            "units_2",
            "activation",
            "optimizer_name",
            "learning_rate",
        ]
    }
    model = build_model(**model_parameters)

    # EarlyStopping often replaces direct epoch tuning in real work. epochs is
    # an upper limit; training stops sooner if validation loss stops improving.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=config["max_epochs"],
        batch_size=config["batch_size"],
        callbacks=[early_stopping],
        verbose=0,
    )

    _, validation_accuracy = model.evaluate(
        X_validation,
        y_validation,
        verbose=0,
    )
    epochs_used = len(history.history["loss"])
    batches_per_epoch = math.ceil(len(X_train) / config["batch_size"])

    trial_results.append(
        (trial_number, validation_accuracy, epochs_used, batches_per_epoch)
    )
    print(
        f"Finished trial {trial_number}: val_accuracy={validation_accuracy:.3f}, "
        f"epochs_used={epochs_used}. "
        f"Trials left: {len(candidate_configs) - trial_number}",
        flush=True,
    )

    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_config = config.copy()


print("Manual tuning trial results:")
for trial_number, accuracy, epochs_used, batches_per_epoch in trial_results:
    print(
        f"Trial {trial_number}: val_accuracy={accuracy:.3f}, "
        f"epochs_used={epochs_used}, batches_per_epoch={batches_per_epoch}"
    )

print("\nBest hyperparameters:")
for name, value in best_config.items():
    print(f"{name}: {value}")


# Refit after tuning. The scaler is now allowed to learn from all development
# rows because hyperparameter selection is finished and the test set stays held out.
final_scaler = StandardScaler()
X_development_scaled = final_scaler.fit_transform(X_development)
X_test_scaled = final_scaler.transform(X_test)

keras.backend.clear_session()
final_model_parameters = {
    key: best_config[key]
    for key in [
        "units_1",
        "units_2",
        "activation",
        "optimizer_name",
        "learning_rate",
    ]
}
final_model = build_model(**final_model_parameters)
print("\nTraining the final model with the best settings...", flush=True)
final_model.fit(
    X_development_scaled,
    y_development,
    epochs=best_config["max_epochs"],
    batch_size=best_config["batch_size"],
    verbose=0,
)

y_probability = final_model.predict(X_test_scaled, verbose=0).ravel()
y_pred = (y_probability >= 0.5).astype(int)
test_loss, test_accuracy = final_model.evaluate(X_test_scaled, y_test, verbose=0)

print("\nTest binary cross-entropy:", round(test_loss, 3))
print("Test accuracy:", round(test_accuracy, 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
