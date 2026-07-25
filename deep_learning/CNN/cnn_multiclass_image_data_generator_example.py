import os
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from _digits_directory_data import create_digits_directory_dataset, plot_learning_curves


# Attribute access avoids Pylance's false missing-import warning for the
# dynamically generated tensorflow.keras submodule.
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
load_img = tf.keras.utils.load_img


tf.get_logger().setLevel("ERROR")
device_name = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print(f"TensorFlow {tf.__version__} is running on {device_name}.", flush=True)

BATCH_SIZE = 20
IMAGE_SIZE = (32, 32)
EPOCHS = 6

temporary_data = tempfile.TemporaryDirectory(prefix="cnn_multiclass_digits_")
train_directory, validation_directory = create_digits_directory_dataset(
    temporary_data.name,
    included_digits=[0, 1, 2, 3, 4],
    train_per_class=100,
    validation_per_class=25,
    image_size=IMAGE_SIZE,
)


# ImageDataGenerator is deprecated but intentionally used for this note example.
training_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.05,
    height_shift_range=0.05,
)
validation_generator = ImageDataGenerator(rescale=1.0 / 255)

training_iterator = training_generator.flow_from_directory(
    str(train_directory),
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    # categorical creates one-hot targets such as [0, 0, 1, 0, 0].
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)
validation_iterator = validation_generator.flow_from_directory(
    str(validation_directory),
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
)

number_of_classes = len(training_iterator.class_indices)
class_names = list(training_iterator.class_indices)


model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(*IMAGE_SIZE, 1)),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        # Dropout randomly disables 20% of values only during training.
        tf.keras.layers.Dropout(0.20),
        # One softmax output is required for every class directory.
        tf.keras.layers.Dense(number_of_classes, activation="softmax"),
    ],
    name="five_digit_multiclass_cnn",
)

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
)

print("\nTraining five-class CNN...")
history = model.fit(
    training_iterator,
    epochs=EPOCHS,
    validation_data=validation_iterator,
    verbose=2,
)


validation_iterator.reset()
test_loss, test_accuracy = model.evaluate(validation_iterator, verbose=0)
validation_iterator.reset()
y_probability = model.predict(validation_iterator, verbose=0)
y_pred = np.argmax(y_probability, axis=1)
y_actual = validation_iterator.classes

print("\nUnseen validation loss:", round(test_loss, 3))
print("Unseen validation accuracy:", round(test_accuracy, 3))
print("\nClassification report:")
print(classification_report(y_actual, y_pred, target_names=class_names, digits=3))

print("Predictions for five unseen image files:")
for index in range(5):
    confidence = y_probability[index, y_pred[index]]
    print(
        f"actual={class_names[y_actual[index]]} "
        f"predicted={class_names[y_pred[index]]} confidence={confidence:.3f}"
    )


plot_learning_curves(history, "Five-class CNN")

test_confusion_matrix = confusion_matrix(y_actual, y_pred)
figure, axis = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    test_confusion_matrix,
    display_labels=class_names,
).plot(ax=axis, cmap="Greens", colorbar=False)
axis.set_title("Five-class CNN on unseen images")
figure.tight_layout()


# Display ten physical validation image files with decoded CNN predictions.
figure, axes = plt.subplots(2, 5, figsize=(11, 5))
for index, axis in enumerate(axes.flat):
    image = load_img(validation_iterator.filepaths[index], color_mode="grayscale")
    is_correct = y_actual[index] == y_pred[index]
    axis.imshow(image, cmap="gray")
    axis.set_title(
        f"actual: {class_names[y_actual[index]]}\n"
        f"pred: {class_names[y_pred[index]]}",
        color="green" if is_correct else "red",
        fontsize=9,
    )
    axis.axis("off")

figure.tight_layout()

print("\nDisplaying learning curves, confusion matrix, and unseen images...")
plt.show()
temporary_data.cleanup()
