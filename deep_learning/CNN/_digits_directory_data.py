from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits


def create_digits_directory_dataset(
    root_directory,
    included_digits,
    train_per_class,
    validation_per_class,
    image_size=(32, 32),
    random_state=42,
):
    """Create temporary class directories compatible with flow_from_directory."""
    root_directory = Path(root_directory)
    train_directory = root_directory / "train"
    validation_directory = root_directory / "validation"

    digits = load_digits()
    rng = np.random.default_rng(random_state)

    for digit in included_digits:
        class_name = f"digit_{digit}"
        train_class_directory = train_directory / class_name
        validation_class_directory = validation_directory / class_name
        train_class_directory.mkdir(parents=True, exist_ok=True)
        validation_class_directory.mkdir(parents=True, exist_ok=True)

        class_indices = np.flatnonzero(digits.target == digit)
        rng.shuffle(class_indices)
        required_images = train_per_class + validation_per_class
        selected_indices = class_indices[:required_images]

        for position, dataset_index in enumerate(selected_indices):
            # sklearn pixel values are 0..16. PNG files conventionally store
            # grayscale intensity as 0..255, so scale before saving.
            pixel_values = (digits.images[dataset_index] / 16.0 * 255).astype(
                np.uint8
            )
            image = Image.fromarray(pixel_values, mode="L")
            image = image.resize(image_size, Image.Resampling.BICUBIC)

            if position < train_per_class:
                destination = train_class_directory
            else:
                destination = validation_class_directory

            image.save(destination / f"sample_{dataset_index}.png")

    return train_directory, validation_directory


def plot_learning_curves(history, title):
    """Create training/validation accuracy and loss graphs for display."""
    epochs = range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.history["accuracy"], label="Training")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[0].set(title=f"{title} accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history.history["loss"], label="Training")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set(title=f"{title} loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    return figure
