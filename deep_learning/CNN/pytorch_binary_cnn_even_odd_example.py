"""Classify handwritten digits as even or odd with a PyTorch CNN."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 7
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} is using: {device}")


# ============================================================
# 1. Turn the digits dataset into a binary image problem
# ============================================================

digits = load_digits()
X = (digits.images / 16.0).astype(np.float32)[:, np.newaxis, :, :]

# Modulo 2 converts each original digit label into:
# 0 = even (0, 2, 4, 6, 8), 1 = odd (1, 3, 5, 7, 9).
y = (digits.target % 2).astype(np.float32)

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
    data = TensorDataset(
        torch.from_numpy(images),
        torch.from_numpy(labels).reshape(-1, 1),
    )
    return DataLoader(data, batch_size=64, shuffle=shuffle)


train_loader = make_loader(X_train, y_train, shuffle=True)
validation_loader = make_loader(X_validation, y_validation, shuffle=False)
test_loader = make_loader(X_test, y_test, shuffle=False)


# ============================================================
# 2. Build a binary CNN with batch normalization
# ============================================================

class EvenOddCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            # BatchNorm2d normalizes each feature channel within a mini-batch.
            # It can stabilize training and permit larger learning rates.
            nn.BatchNorm2d(12),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(24 * 2 * 2, 24),
            nn.LeakyReLU(negative_slope=0.1),
            # One raw logit represents the odd-class score.
            nn.Linear(24, 1),
        )

    def forward(self, images):
        return self.network(images)


model = EvenOddCNN().to(device)
print("\nModel architecture:\n", model)

loss_function = nn.BCEWithLogitsLoss()

# SGD uses one global learning rate. momentum carries part of the previous
# update forward, helping optimization move consistently through noisy gradients.
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.03,
    momentum=0.9,
    weight_decay=0.0001,
)


def run_epoch(loader, training):
    model.train(mode=training)
    total_loss = 0.0
    correct = 0
    count = 0

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss = loss_function(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            predictions = (torch.sigmoid(logits) >= 0.5).float()
            total_loss += loss.item() * targets.size(0)
            correct += (predictions == targets).sum().item()
            count += targets.size(0)

    return total_loss / count, correct / count


EPOCHS = 12
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
# 3. Evaluate probabilities on unseen images
# ============================================================

model.eval()
test_probability_parts = []
with torch.inference_mode():
    for images, _ in test_loader:
        logits = model(images.to(device))
        test_probability_parts.append(torch.sigmoid(logits).cpu())

test_probabilities = torch.cat(test_probability_parts).numpy().ravel()
test_predictions = (test_probabilities >= 0.5).astype(int)

print("\nClassification report:")
print(
    classification_report(
        y_test.astype(int),
        test_predictions,
        target_names=["even", "odd"],
    )
)

print("First five unseen predictions:")
for actual, predicted, probability in zip(
    y_test[:5].astype(int),
    test_predictions[:5],
    test_probabilities[:5],
):
    print(
        f"actual={['even', 'odd'][actual]:4s} | "
        f"predicted={['even', 'odd'][predicted]:4s} | "
        f"P(odd)={probability:.3f}"
    )


# ============================================================
# 4. Display complementary evaluation graphs
# ============================================================

epochs = range(1, EPOCHS + 1)
figure, axes = plt.subplots(2, 2, figsize=(12, 9))
axes[0, 0].plot(epochs, history["train_loss"], label="Training")
axes[0, 0].plot(epochs, history["val_loss"], label="Validation")
axes[0, 0].set(title="Even/odd CNN loss", xlabel="Epoch", ylabel="Loss")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.25)

axes[0, 1].plot(epochs, history["train_acc"], label="Training")
axes[0, 1].plot(epochs, history["val_acc"], label="Validation")
axes[0, 1].set(title="Even/odd CNN accuracy", xlabel="Epoch", ylabel="Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.25)

ConfusionMatrixDisplay.from_predictions(
    y_test.astype(int),
    test_predictions,
    display_labels=["even", "odd"],
    cmap="Oranges",
    ax=axes[1, 0],
    colorbar=False,
)
axes[1, 0].set_title("Binary confusion matrix")

# ROC plots sensitivity against false-positive rate over every possible
# probability threshold. A curve near the top-left indicates strong separation.
RocCurveDisplay.from_predictions(
    y_test.astype(int),
    test_probabilities,
    name="PyTorch CNN",
    ax=axes[1, 1],
)
axes[1, 1].set_title("Odd-class ROC curve")
axes[1, 1].grid(alpha=0.25)
figure.tight_layout()
plt.show()
