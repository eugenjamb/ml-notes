"""Recognize handwritten digits with a multiclass PyTorch CNN."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} is using: {device}")


# ============================================================
# 1. Load image data from scikit-learn
# ============================================================

digits = load_digits()

# sklearn supplies grayscale images with shape (samples, height, width) and
# pixel values from 0 to 16. Dividing by 16 scales pixels to 0..1.
X = (digits.images / 16.0).astype(np.float32)
y = digits.target.astype(np.int64)

# Conv2d requires NCHW layout: samples, channels, height, width. These are
# grayscale images, so np.newaxis inserts one channel.
X = X[:, np.newaxis, :, :]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)


def make_loader(images, labels, shuffle):
    dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))
    # Each batch has shape (64, 1, 8, 8), except possibly the final batch.
    return DataLoader(dataset, batch_size=64, shuffle=shuffle)


train_loader = make_loader(X_train, y_train, shuffle=True)
validation_loader = make_loader(X_validation, y_validation, shuffle=False)
test_loader = make_loader(X_test, y_test, shuffle=False)


# ============================================================
# 2. Define convolutional and dense layers
# ============================================================

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # in_channels=1 accepts grayscale input. out_channels=16 learns 16
            # filters. kernel_size=3 inspects 3x3 neighborhoods, while padding=1
            # preserves the 8x8 width and height.
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # MaxPool2d keeps the strongest value in each 2x2 region, reducing
            # 8x8 feature maps to 4x4.
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # The second pool changes 32 x 4 x 4 into 32 x 2 x 2.
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            # Flatten keeps the batch dimension and converts each image's
            # feature maps into 32 * 2 * 2 = 128 values.
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            # Ten logits correspond to digit classes 0 through 9.
            nn.Linear(64, 10),
        )

    def forward(self, images):
        features = self.feature_extractor(images)
        return self.classifier(features)


model = DigitCNN().to(device)
print("\nModel architecture:\n", model)

# CrossEntropyLoss combines log-softmax with multiclass negative log likelihood.
# It expects raw logits and integer labels, so no Softmax belongs in forward().
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ============================================================
# 3. Train the CNN
# ============================================================

def run_epoch(loader, training):
    model.train(mode=training)
    total_loss = 0.0
    correct = 0
    count = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss = loss_function(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            count += labels.size(0)

    return total_loss / count, correct / count


EPOCHS = 15
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
for epoch in range(1, EPOCHS + 1):
    train_loss, train_accuracy = run_epoch(train_loader, training=True)
    validation_loss, validation_accuracy = run_epoch(
        validation_loader,
        training=False,
    )
    history["train_loss"].append(train_loss)
    history["val_loss"].append(validation_loss)
    history["train_acc"].append(train_accuracy)
    history["val_acc"].append(validation_accuracy)

    if epoch == 1 or epoch % 3 == 0:
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_loss:.4f}, val loss={validation_loss:.4f} | "
            f"train acc={train_accuracy:.3f}, val acc={validation_accuracy:.3f}"
        )


# ============================================================
# 4. Predict the untouched test images
# ============================================================

model.eval()
probability_batches = []
with torch.inference_mode():
    for images, _ in test_loader:
        logits = model(images.to(device))
        probability_batches.append(torch.softmax(logits, dim=1).cpu())

probabilities = torch.cat(probability_batches).numpy()
predictions = probabilities.argmax(axis=1)

print("\nClassification report:")
print(classification_report(y_test, predictions, digits=4))
print("First ten actual labels:   ", y_test[:10])
print("First ten predictions:    ", predictions[:10])
print("First ten confidence scores:", np.round(probabilities.max(axis=1)[:10], 3))


# ============================================================
# 5. Plot learning curves, confusion matrix, and unseen images
# ============================================================

epochs = range(1, EPOCHS + 1)
figure, axes = plt.subplots(2, 2, figsize=(13, 10))
axes[0, 0].plot(epochs, history["train_loss"], label="Training")
axes[0, 0].plot(epochs, history["val_loss"], label="Validation")
axes[0, 0].set(title="CNN loss", xlabel="Epoch", ylabel="Loss")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.25)

axes[0, 1].plot(epochs, history["train_acc"], label="Training")
axes[0, 1].plot(epochs, history["val_acc"], label="Validation")
axes[0, 1].set(title="CNN accuracy", xlabel="Epoch", ylabel="Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.25)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    cmap="Blues",
    ax=axes[1, 0],
    colorbar=False,
)
axes[1, 0].set_title("Digit confusion matrix")

# Display one genuinely unseen test image and the CNN's decision.
sample_index = 0
axes[1, 1].imshow(X_test[sample_index, 0], cmap="gray")
axes[1, 1].set_title(
    f"Actual: {y_test[sample_index]} | Predicted: {predictions[sample_index]}\n"
    f"Confidence: {probabilities[sample_index].max():.3f}"
)
axes[1, 1].axis("off")
figure.tight_layout()
plt.show()

