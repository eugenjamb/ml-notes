"""Multiclass ANN with PyTorch, AdamW, and CrossEntropyLoss."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_wine
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} is using: {device}")


# ============================================================
# 1. Prepare a three-class dataset
# ============================================================

wine = load_wine()
X = wine.data.astype(np.float32)

# CrossEntropyLoss requires integer class indices (torch.long), not one-hot
# encoded targets. The labels here are 0, 1, and 2.
y = wine.target.astype(np.int64)

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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_validation = scaler.transform(X_validation).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)


def make_loader(features, targets, shuffle):
    data = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    return DataLoader(data, batch_size=16, shuffle=shuffle)


train_loader = make_loader(X_train, y_train, shuffle=True)
validation_loader = make_loader(X_validation, y_validation, shuffle=False)
test_loader = make_loader(X_test, y_test, shuffle=False)


# ============================================================
# 2. Build a multiclass network
# ============================================================

class WineANN(nn.Module):
    def __init__(self, input_features, number_of_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            # GELU is a smooth nonlinear activation often used in modern neural
            # networks. Unlike ReLU, it does not abruptly set all negatives to 0.
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(64, 32),
            nn.GELU(),
            # One output logit per class. CrossEntropyLoss applies softmax
            # internally, so the model must return raw logits.
            nn.Linear(32, number_of_classes),
        )

    def forward(self, features):
        return self.network(features)


model = WineANN(X_train.shape[1], len(wine.target_names)).to(device)
print("\nModel architecture:\n", model)

# CrossEntropyLoss compares class logits against integer class labels.
loss_function = nn.CrossEntropyLoss()

# AdamW separates weight decay from Adam's adaptive gradient update. This is a
# commonly used optimizer when explicit weight regularization is wanted.
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.002,
    weight_decay=0.001,
)


def run_epoch(loader, training):
    model.train(mode=training)
    total_loss = 0.0
    correct = 0
    count = 0

    with torch.set_grad_enabled(training):
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            if training:
                optimizer.zero_grad()

            logits = model(features)
            loss = loss_function(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * targets.size(0)
            correct += (predictions == targets).sum().item()
            count += targets.size(0)

    return total_loss / count, correct / count


EPOCHS = 30
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

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_loss:.4f}, val loss={validation_loss:.4f} | "
            f"train acc={train_accuracy:.3f}, val acc={validation_accuracy:.3f}"
        )


# ============================================================
# 3. Convert test logits to probabilities and class predictions
# ============================================================

model.eval()
probability_batches = []
with torch.inference_mode():
    for features, _ in test_loader:
        logits = model(features.to(device))
        # softmax(dim=1) converts each row of class logits into probabilities
        # that sum to 1. argmax selects the most probable class.
        probability_batches.append(torch.softmax(logits, dim=1).cpu())

probabilities = torch.cat(probability_batches).numpy()
predictions = probabilities.argmax(axis=1)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=wine.target_names,
    )
)

print("First five unseen predictions:")
for actual, predicted, confidence in zip(
    y_test[:5],
    predictions[:5],
    probabilities.max(axis=1)[:5],
):
    print(
        f"actual={wine.target_names[actual]:7s} | "
        f"predicted={wine.target_names[predicted]:7s} | "
        f"confidence={confidence:.3f}"
    )


# ============================================================
# 4. Graph training history and test errors
# ============================================================

epochs = range(1, EPOCHS + 1)
figure, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(epochs, history["train_loss"], label="Training")
axes[0].plot(epochs, history["val_loss"], label="Validation")
axes[0].set(title="Multiclass ANN loss", xlabel="Epoch", ylabel="Loss")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(epochs, history["train_acc"], label="Training")
axes[1].plot(epochs, history["val_acc"], label="Validation")
axes[1].set(title="Multiclass ANN accuracy", xlabel="Epoch", ylabel="Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.25)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=wine.target_names,
    cmap="YlGn",
    ax=axes[2],
    colorbar=False,
)
axes[2].set_title("Wine test predictions")
figure.tight_layout()
plt.show()

