import os
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from _digits_directory_data import (
    create_digits_directory_dataset,
    plot_learning_curves,
)


# Access these through tf.keras because TensorFlow creates the tensorflow.keras
# package dynamically, which can make Pylance report its submodule as missing.
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
img_to_array = tf.keras.utils.img_to_array
load_img = tf.keras.utils.load_img


# ImageDataGenerator is kept here because this example specifically teaches its
# API. TensorFlow marks it deprecated; new projects usually use
# tf.keras.utils.image_dataset_from_directory and augmentation layers instead.
tf.get_logger().setLevel("ERROR")
device_name = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print(f"TensorFlow {tf.__version__} is running on {device_name}.", flush=True)

BATCH_SIZE = 16
IMAGE_SIZE = (32, 32)
EPOCHS = 5

# flow_from_directory needs physical class folders. TemporaryDirectory keeps
# generated learning data outside the project and removes it after this run.
temporary_data = tempfile.TemporaryDirectory(prefix="cnn_binary_digits_")
train_directory, validation_directory = create_digits_directory_dataset(
    root_directory=temporary_data.name,
    included_digits=[0, 1],
    train_per_class=120,
    validation_per_class=35,
    image_size=IMAGE_SIZE,
)


print("\nLoading and augmenting training images...")
training_data_generator = ImageDataGenerator(
    # Convert PNG pixel values from 0..255 to neural-network values from 0..1.
    rescale=1.0 / 255,
    # Random augmentation creates slightly different training images each epoch.
    zoom_range=0.15,
    rotation_range=12,
    width_shift_range=0.08,
    height_shift_range=0.08,
)

training_iterator = training_data_generator.flow_from_directory(
    str(train_directory),
    target_size=IMAGE_SIZE,
    # grayscale produces one channel, so model input shape ends with 1.
    color_mode="grayscale",
    # binary returns one 0/1 label and matches BinaryCrossentropy.
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)


print("\nLoading unseen validation images...")
validation_data_generator = ImageDataGenerator(rescale=1.0 / 255)
validation_iterator = validation_data_generator.flow_from_directory(
    str(validation_directory),
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    class_mode="binary",
    batch_size=BATCH_SIZE,
    # Keep file order stable so predictions align with .classes and .filepaths.
    shuffle=False,
)


print("\nBuilding binary CNN...")
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(*IMAGE_SIZE, 1)),
        # filters=8 learns eight local patterns with a 3 x 3 sliding kernel.
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation="relu"),
        # Max pooling reduces width/height and keeps the strongest local response.
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=16, activation="relu"),
        # One sigmoid output is correct for two classes with binary labels.
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ],
    name="binary_digit_cnn",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)
model.summary()


print("\nTraining binary CNN...")
history = model.fit(
    training_iterator,
    epochs=EPOCHS,
    validation_data=validation_iterator,
    # One line per epoch avoids a noisy progress bar for every batch.
    verbose=2,
)


# Evaluate all held-out images that augmentation/training never used.
validation_iterator.reset()
test_metrics = model.evaluate(validation_iterator, verbose=0, return_dict=True)
validation_iterator.reset()
y_probability = model.predict(validation_iterator, verbose=0).ravel()
y_pred = (y_probability >= 0.5).astype(int)
y_actual = validation_iterator.classes.astype(int)

class_names = list(validation_iterator.class_indices)
print("\nUnseen validation metrics:")
for metric_name, value in test_metrics.items():
    print(f"{metric_name}: {value:.3f}")
print("\nClassification report:")
print(classification_report(y_actual, y_pred, target_names=class_names, digits=3))


# Predict one individual unseen image, as would happen after deployment.
unseen_image_path = validation_iterator.filepaths[0]
unseen_image = load_img(
    unseen_image_path,
    color_mode="grayscale",
    target_size=IMAGE_SIZE,
)
unseen_array = img_to_array(unseen_image) / 255.0
unseen_probability = float(
    model.predict(np.expand_dims(unseen_array, axis=0), verbose=0)[0, 0]
)
unseen_prediction = int(unseen_probability >= 0.5)
unseen_actual = y_actual[0]

print("\nOne unseen image prediction:")
print("File:", unseen_image_path)
print("Actual:", class_names[unseen_actual])
print("Predicted:", class_names[unseen_prediction])
print("Probability of digit_1:", round(unseen_probability, 3))


plot_learning_curves(history, "Binary CNN")

figure, axes = plt.subplots(1, 2, figsize=(10, 4))
test_confusion_matrix = confusion_matrix(y_actual, y_pred)
ConfusionMatrixDisplay(
    test_confusion_matrix,
    display_labels=class_names,
).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Unseen validation confusion matrix")

axes[1].imshow(unseen_array.squeeze(), cmap="gray")
axes[1].set_title(
    f"Actual: {class_names[unseen_actual]}\n"
    f"Predicted: {class_names[unseen_prediction]}"
)
axes[1].axis("off")
figure.tight_layout()

print("\nDisplaying learning curves and unseen-image diagnostics...")
plt.show()
temporary_data.cleanup()
