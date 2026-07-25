import os
import tempfile
import warnings
from pathlib import Path

# Request less output while keeping oneDNN's Intel/AMD CPU optimizations.
# TensorFlow 2.21 may still print two oneDNN lines before logging initializes.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras_tuner as kt
import keras
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


tf.get_logger().setLevel("ERROR")
device_name = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print(f"TensorFlow {tf.__version__} is running on {device_name}.", flush=True)

# Hyperband promotes model weights between rounds but intentionally creates a
# fresh optimizer. Keras may warn that old optimizer variables were skipped;
# this expected message does not affect the promoted model weights.
warnings.filterwarnings(
    "ignore",
    message="Skipping variable loading for optimizer.*",
    category=UserWarning,
)


# ============================================================
# Resource-Efficient ANN Tuning With KerasTuner Hyperband
# ============================================================
#
# Hyperparameters we can tune for an ANN include:
# - number of hidden layers and units in each hidden layer
# - activations such as relu or tanh
# - optimizer and learning rate
# - batch_size (the number of batches is calculated from it)
# - training duration, dropout, L1/L2, and other regularization
#
# Hyperband is practical when training is expensive. It starts many candidate
# models with small epoch budgets, stops weaker candidates, and gives more epochs
# to promising candidates. EarlyStopping adds another limit when progress stalls.
#
# The output and loss are not searched here. Three integer classes require three
# softmax neurons and sparse categorical cross-entropy.
#
# Install KerasTuner and its TensorBoard integration dependency with:
# py -3 -m pip install --upgrade keras-tuner tensorboard


keras.utils.set_random_seed(42)

data = load_wine()
X_development, X_test, y_development, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.20,
    stratify=data.target,
    random_state=42,
)
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
X_test = scaler.transform(X_test)


class WineANNHyperModel(kt.HyperModel):
    def build(self, hp):
        activation = hp.Choice("activation", ["relu", "tanh"])
        number_of_hidden_layers = hp.Int(
            "hidden_layers",
            min_value=1,
            max_value=2,
        )

        model = keras.Sequential(
            [keras.layers.Input(shape=(X_train.shape[1],))]
        )

        # Each layer has a separately named units hyperparameter. Trials with
        # more layers therefore gain additional architecture choices.
        for layer_number in range(number_of_hidden_layers):
            model.add(
                keras.layers.Dense(
                    units=hp.Int(
                        f"units_{layer_number + 1}",
                        min_value=16,
                        max_value=48,
                        step=16,
                    ),
                    activation=activation,
                )
            )
            model.add(
                keras.layers.Dropout(
                    rate=hp.Float(
                        f"dropout_{layer_number + 1}",
                        min_value=0.0,
                        max_value=0.4,
                        step=0.2,
                    )
                )
            )

        model.add(keras.layers.Dense(3, activation="softmax"))

        learning_rate = hp.Float(
            "learning_rate",
            min_value=1e-4,
            max_value=1e-2,
            sampling="log",
        )
        optimizer_name = hp.Choice("optimizer", ["adam", "rmsprop"])
        optimizer_options = {
            "adam": keras.optimizers.Adam(learning_rate=learning_rate),
            "rmsprop": keras.optimizers.RMSprop(learning_rate=learning_rate),
        }

        model.compile(
            optimizer=optimizer_options[optimizer_name],
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        # Hyperband supplies epochs and initial_epoch itself. This override tunes
        # only batch_size and leaves epoch allocation to the Hyperband algorithm.
        kwargs["batch_size"] = hp.Choice("batch_size", [16, 32])
        return model.fit(*args, **kwargs)


class WindowsSafeHyperband(kt.Hyperband):
    """Create safe Windows checkpoints and print concise trial progress."""

    def __init__(self, *args, expected_trials, **kwargs):
        self.expected_trials = expected_trials
        self.completed_trials = 0
        super().__init__(*args, **kwargs)

    def get_trial_dir(self, trial_id):
        # KerasTuner normally creates this directory through TensorFlow's
        # filesystem API. With some TensorFlow/KerasTuner combinations on
        # native Windows, that API can report success before the folder is
        # available to h5py. pathlib performs an additional native mkdir.
        trial_directory = Path(super().get_trial_dir(trial_id))
        trial_directory.mkdir(parents=True, exist_ok=True)
        return str(trial_directory)

    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        self.completed_trials += 1
        trials_left = max(self.expected_trials - self.completed_trials, 0)
        print(
            f"Hyperband trial {self.completed_trials}/{self.expected_trials} "
            f"finished. Estimated trials left: {trials_left}",
            flush=True,
        )


# Hyperband needs temporary checkpoints to promote good models between rounds.
# They are stored outside the project and deleted at the end of this script.
temporary_tuner_directory = tempfile.TemporaryDirectory(prefix="ann_hyperband_")
expected_hyperband_trials = 10


tuner = WindowsSafeHyperband(
    hypermodel=WineANNHyperModel(),
    objective=kt.Objective("val_accuracy", direction="max"),
    # max_epochs is the largest resource allocation for a surviving model.
    max_epochs=6,
    # factor=3 controls how aggressively candidates are reduced between rounds.
    factor=3,
    # One iteration is enough for this small educational example. More
    # iterations search more thoroughly but require more computation.
    hyperband_iterations=1,
    seed=42,
    directory=temporary_tuner_directory.name,
    project_name="ann_hyperband",
    overwrite=True,
    expected_trials=expected_hyperband_trials,
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

print(
    "Starting fast Hyperband with about 10 short trials.",
    flush=True,
)
tuner.search(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    callbacks=[early_stopping],
    # verbose=0 hides individual epochs; the tuner prints one line per trial.
    verbose=0,
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build a clean final model from the winning hyperparameters. Loading a tuner
# checkpoint can emit optimizer-state warnings because only model weights need
# to be promoted between Hyperband rounds.
best_model = tuner.hypermodel.build(best_hp)

# The search is finished, so remove all temporary checkpoints now.
temporary_tuner_directory.cleanup()

X_final_train = np.concatenate([X_train, X_validation])
y_final_train = np.concatenate([y_train, y_validation])
print("Training one final model with the winning settings...", flush=True)
best_model.fit(
    X_final_train,
    y_final_train,
    # Use the full small budget for the final model after Hyperband selection.
    epochs=6,
    batch_size=best_hp.get("batch_size"),
    verbose=0,
)

print("Best Hyperband hyperparameters:")
for name, value in best_hp.values.items():
    print(f"{name}: {value}")

y_probability = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_probability, axis=1)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)

print("\nTest sparse categorical cross-entropy:", round(test_loss, 3))
print("Test accuracy:", round(test_accuracy, 3))
print("First three probability rows:")
print(np.round(y_probability[:3], 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print("Temporary Hyperband files removed.")
