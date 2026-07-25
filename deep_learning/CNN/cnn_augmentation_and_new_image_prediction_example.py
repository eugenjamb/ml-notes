import os
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from _digits_directory_data import create_digits_directory_dataset, plot_learning_curves


# Attribute access avoids Pylance's false missing-import warning for the
# dynamically generated tensorflow.keras submodule.
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
img_to_array = tf.keras.utils.img_to_array
load_img = tf.keras.utils.load_img


tf.get_logger().setLevel("ERROR")
device_name = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print(f"TensorFlow {tf.__version__} is running on {device_name}.", flush=True)

BATCH_SIZE = 32
IMAGE_SIZE = (32, 32)
EPOCHS = 8

temporary_data = tempfile.TemporaryDirectory(prefix="cnn_augmented_digits_")
train_directory, validation_directory = create_digits_directory_dataset(
    temporary_data.name,
    included_digits=range(10),
    train_per_class=100,
    validation_per_class=20,
    image_size=IMAGE_SIZE,
)


# These transformations create new variants in memory; they do not overwrite
# the original PNG files. Keep ranges realistic so a digit remains recognizable.
training_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=8,
    zoom_range=0.10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.03,
)
validation_generator = ImageDataGenerator(rescale=1.0 / 255)

training_iterator = training_generator.flow_from_directory(
    str(train_directory),
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
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

class_names = list(training_iterator.class_indices)


model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(*IMAGE_SIZE, 1)),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        # Flatten preserves where each learned feature occurs. Spatial position
        # matters for distinguishing small handwritten digit shapes.
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ],
    name="augmented_ten_digit_cnn",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

print("\nTraining ten-class CNN with newly augmented images...")
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
print(
    classification_report(
        y_actual,
        y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
)


# Load one original training image and ask ImageDataGenerator to create six new
# transformed versions. The returned images are already rescaled to 0..1.
source_image_path = training_iterator.filepaths[0]
source_image = load_img(
    source_image_path,
    color_mode="grayscale",
    target_size=IMAGE_SIZE,
)
source_array = img_to_array(source_image)
augmentation_iterator = training_generator.flow(
    np.expand_dims(source_array, axis=0),
    batch_size=1,
    shuffle=False,
    seed=7,
)
augmented_images = np.concatenate(
    [next(augmentation_iterator) for _ in range(6)],
    axis=0,
)

# Predict each newly created image, not only images read from validation folders.
augmented_probabilities = model.predict(augmented_images, verbose=0)
augmented_predictions = np.argmax(augmented_probabilities, axis=1)

figure, axes = plt.subplots(2, 3, figsize=(8, 6))
for index, axis in enumerate(axes.flat):
    predicted_class = class_names[augmented_predictions[index]]
    confidence = np.max(augmented_probabilities[index])
    axis.imshow(augmented_images[index].squeeze(), cmap="gray")
    axis.set_title(f"{predicted_class}, confidence {confidence:.2f}")
    axis.axis("off")

figure.suptitle("New augmented images and CNN predictions")
figure.tight_layout()

plot_learning_curves(history, "Augmented ten-class CNN")

print("\nSource image:", source_image_path)
print("Predictions for six newly created augmented images:")
for index, predicted_index in enumerate(augmented_predictions, start=1):
    print(
        f"augmentation {index}: {class_names[predicted_index]}, "
        f"confidence={np.max(augmented_probabilities[index - 1]):.3f}"
    )

print("\nDisplaying augmented-image predictions and learning curves...")
plt.show()
temporary_data.cleanup()
