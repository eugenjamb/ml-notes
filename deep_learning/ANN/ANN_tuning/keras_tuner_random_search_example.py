import os
import tempfile

# Request less TensorFlow output while keeping oneDNN CPU optimizations enabled.
# TensorFlow 2.21 may still print two oneDNN lines before logging initializes.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras_tuner as kt
import keras
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


tf.get_logger().setLevel("ERROR")
device_name = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print(f"TensorFlow {tf.__version__} is running on {device_name}.", flush=True)


# ============================================================
# Automatic ANN Tuning With KerasTuner RandomSearch
# ============================================================
#
# ANN hyperparameters commonly tuned in real projects:
# - hidden-layer count and units per layer
# - hidden activation functions
# - optimizer and learning rate
# - batch_size, which determines the number of batches per epoch
# - maximum epochs plus an EarlyStopping policy
# - dropout rate, weight decay, L1/L2, and weight initialization
#
# batch_size is tuned; number of batches is derived from data_size / batch_size.
# The loss is fixed because binary labels and one sigmoid output require binary
# cross-entropy. A loss should match the problem, not be treated as an arbitrary
# performance switch.
#
# Install the tuning and TensorBoard integration packages with:
# py -3 -m pip install --upgrade keras-tuner tensorboard


keras.utils.set_random_seed(42)

data = load_breast_cancer()
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


class ANNHyperModel(kt.HyperModel):
    """Tell KerasTuner how to build and train each candidate ANN."""

    def build(self, hp):
        # hp.Choice samples one listed value for each trial.
        activation = hp.Choice("activation", values=["relu", "tanh"])
        optimizer_name = hp.Choice(
            "optimizer",
            values=["adam", "rmsprop", "sgd"],
        )

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(X_train.shape[1],)),
                keras.layers.Dense(
                    # hp.Int samples 16, 32, or 48 neurons.
                    units=hp.Int("units_1", 16, 48, step=16),
                    activation=activation,
                ),
                keras.layers.Dense(
                    units=hp.Int("units_2", 8, 24, step=8),
                    activation=activation,
                ),
                # hp.Float tunes dropout regularization from 0% to 40%.
                keras.layers.Dropout(
                    rate=hp.Float("dropout", 0.0, 0.4, step=0.2)
                ),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        # Log sampling explores multiplicative scales such as 0.0001, 0.001,
        # and 0.01 more appropriately than evenly spaced linear values.
        learning_rate = hp.Float(
            "learning_rate",
            min_value=1e-4,
            max_value=1e-2,
            sampling="log",
        )
        optimizer_options = {
            "adam": keras.optimizers.Adam(learning_rate=learning_rate),
            "rmsprop": keras.optimizers.RMSprop(learning_rate=learning_rate),
            "sgd": keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=0.9,
            ),
        }

        model.compile(
            optimizer=optimizer_options[optimizer_name],
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        # Training hyperparameters belong in fit(), rather than build().
        kwargs["batch_size"] = hp.Choice("batch_size", [32, 64])
        kwargs["epochs"] = hp.Choice("max_epochs", [15, 25])
        return model.fit(*args, **kwargs)


class QuietRandomSearch(kt.RandomSearch):
    """Print one short progress line after each completed trial."""

    def __init__(self, *args, planned_trials, **kwargs):
        self.planned_trials = planned_trials
        self.completed_trials = 0
        super().__init__(*args, **kwargs)

    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        self.completed_trials += 1
        trials_left = max(self.planned_trials - self.completed_trials, 0)
        print(
            f"RandomSearch trial {self.completed_trials}/{self.planned_trials} "
            f"finished. Trials left: {trials_left}",
            flush=True,
        )


# KerasTuner needs checkpoints while searching. TemporaryDirectory puts them
# outside the project and removes them automatically when cleanup() is called.
temporary_tuner_directory = tempfile.TemporaryDirectory(
    prefix="ann_random_search_"
)
number_of_trials = 4

tuner = QuietRandomSearch(
    hypermodel=ANNHyperModel(),
    # val_accuracy is maximized to select the winning trial.
    objective=kt.Objective("val_accuracy", direction="max"),
    # max_trials limits cost. Random search samples this many configurations
    # instead of evaluating every possible combination.
    max_trials=number_of_trials,
    # One execution is faster; increase this to reduce random result variance.
    executions_per_trial=1,
    seed=42,
    directory=temporary_tuner_directory.name,
    project_name="ann_random_search",
    # overwrite=True starts a fresh search instead of resuming saved trials.
    overwrite=True,
    planned_trials=number_of_trials,
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
)

print(
    f"Starting RandomSearch with {number_of_trials} trials.",
    flush=True,
)
tuner.search(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    callbacks=[early_stopping],
    # verbose=0 hides individual epochs; QuietRandomSearch reports each trial.
    verbose=0,
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

# Rebuild the winning architecture instead of loading KerasTuner's optimizer
# checkpoint. This avoids optimizer-state warnings and gives us a clean model.
best_model = tuner.hypermodel.build(best_hp)

# The search is finished, so its temporary checkpoints are no longer needed.
temporary_tuner_directory.cleanup()

X_final_train = np.concatenate([X_train, X_validation])
y_final_train = np.concatenate([y_train, y_validation])
print("Training one final model with the winning settings...", flush=True)
best_model.fit(
    X_final_train,
    y_final_train,
    epochs=best_hp.get("max_epochs"),
    batch_size=best_hp.get("batch_size"),
    verbose=0,
)

print("Best RandomSearch hyperparameters:")
for name, value in best_hp.values.items():
    print(f"{name}: {value}")

y_probability = best_model.predict(X_test, verbose=0).ravel()
y_pred = (y_probability >= 0.5).astype(int)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)

print("\nTest binary cross-entropy:", round(test_loss, 3))
print("Test accuracy:", round(test_accuracy, 3))
print("First five probabilities:", np.round(y_probability[:5], 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print("Temporary RandomSearch files removed.")
