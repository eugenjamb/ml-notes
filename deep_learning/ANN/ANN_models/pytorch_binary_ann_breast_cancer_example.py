"""Binary classification with a fully connected PyTorch neural network."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Reproducible seeds make the split, initial weights, and shuffled batches more
# consistent. Small floating-point differences can still occur across hardware.
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# CUDA is used when available; otherwise PyTorch trains normally on the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} is using: {device}")


# ============================================================
# 1. Load, split, and scale the data
# ============================================================

dataset = load_breast_cancer()
X = dataset.data.astype(np.float32)
y = dataset.target.astype(np.float32)

# First isolate the final test set. stratify preserves the malignant/benign
# proportions, and random_state makes the split repeatable.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Validation data measures generalization during training. Because 25% of the
# remaining 80% is selected, validation receives 20% of the complete dataset.
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

# Fit preprocessing only on training data to prevent information leakage.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_validation = scaler.transform(X_validation).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)


def make_loader(features, targets, batch_size, shuffle):
    """Convert NumPy arrays to mini-batches of PyTorch tensors."""
    tensor_dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(targets).reshape(-1, 1),
    )
    # batch_size controls samples per weight update. shuffle=True changes the
    # training order each epoch; validation/test order should remain stable.
    return DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


BATCH_SIZE = 32
train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
validation_loader = make_loader(
    X_validation,
    y_validation,
    BATCH_SIZE,
    shuffle=False,
)
test_loader = make_loader(X_test, y_test, BATCH_SIZE, shuffle=False)


# ============================================================
# 2. Define the ANN architecture
# ============================================================

class BinaryANN(nn.Module):
    """A feed-forward network that returns one raw binary-class logit."""

    def __init__(self, number_of_features):
        super().__init__()
        self.layers = nn.Sequential(
            # Linear connects every input feature to each of 32 neurons.
            nn.Linear(in_features=number_of_features, out_features=32),
            # ReLU introduces nonlinearity: max(0, x).
            nn.ReLU(),
            # Dropout randomly disables 20% of activations during training to
            # reduce reliance on individual neurons and limit overfitting.
            nn.Dropout(p=0.20),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            # One output is enough for binary classification. It is a raw
            # logit, so no Sigmoid layer is placed inside the model.
            nn.Linear(in_features=16, out_features=1),
        )

    def forward(self, features):
        return self.layers(features)


model = BinaryANN(number_of_features=X_train.shape[1]).to(device)
print("\nModel architecture:\n", model)

# BCEWithLogitsLoss combines a numerically stable sigmoid operation with binary
# cross-entropy. Targets must be float values shaped like the output logits.
loss_function = nn.BCEWithLogitsLoss()

# Adam adapts the update size for each parameter. lr is the learning rate, and
# weight_decay applies L2 regularization to discourage very large weights.
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001,
)


# ============================================================
# 3. Train and validate
# ============================================================

def run_epoch(data_loader, training):
    """Run one complete pass and return average loss and accuracy."""
    model.train(mode=training)
    total_loss = 0.0
    correct = 0
    sample_count = 0

    # Gradients are required only during training. Disabling them for
    # validation reduces memory usage and computation.
    with torch.set_grad_enabled(training):
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            if training:
                # PyTorch accumulates gradients, so clear old values first.
                optimizer.zero_grad()

            logits = model(features)
            loss = loss_function(logits, targets)

            if training:
                loss.backward()  # Calculate gradients by backpropagation.
                optimizer.step()  # Update model weights using those gradients.

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            correct += (predictions == targets).sum().item()
            sample_count += batch_size

    return total_loss / sample_count, correct / sample_count


EPOCHS = 25
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

    # Print occasional summaries instead of logging every mini-batch.
    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_loss:.4f}, val loss={validation_loss:.4f} | "
            f"train acc={train_accuracy:.3f}, val acc={validation_accuracy:.3f}"
        )


# ============================================================
# 4. Predict unseen test data and report results
# ============================================================

model.eval()
test_probabilities = []
with torch.inference_mode():
    for features, _ in test_loader:
        logits = model(features.to(device))
        test_probabilities.extend(torch.sigmoid(logits).cpu().numpy().ravel())

test_probabilities = np.asarray(test_probabilities)
test_predictions = (test_probabilities >= 0.5).astype(int)

print("\nTest accuracy:", round(accuracy_score(y_test, test_predictions), 4))
print("\nClassification report:")
print(
    classification_report(
        y_test.astype(int),
        test_predictions,
        target_names=dataset.target_names,
    )
)

print("First five unseen predictions:")
for actual, predicted, probability in zip(
    y_test[:5].astype(int),
    test_predictions[:5],
    test_probabilities[:5],
):
    print(
        f"actual={dataset.target_names[actual]:9s} | "
        f"predicted={dataset.target_names[predicted]:9s} | "
        f"P(benign)={probability:.3f}"
    )


# ============================================================
# 5. Plot learning curves and the confusion matrix
# ============================================================

epochs = range(1, EPOCHS + 1)
figure, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(epochs, history["train_loss"], label="Training")
axes[0].plot(epochs, history["val_loss"], label="Validation")
axes[0].set(title="Binary ANN loss", xlabel="Epoch", ylabel="Loss")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(epochs, history["train_acc"], label="Training")
axes[1].plot(epochs, history["val_acc"], label="Validation")
axes[1].set(title="Binary ANN accuracy", xlabel="Epoch", ylabel="Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.25)

ConfusionMatrixDisplay.from_predictions(
    y_test.astype(int),
    test_predictions,
    display_labels=dataset.target_names,
    cmap="Blues",
    ax=axes[2],
    colorbar=False,
)
axes[2].set_title("Unseen test predictions")
figure.tight_layout()
plt.show()

