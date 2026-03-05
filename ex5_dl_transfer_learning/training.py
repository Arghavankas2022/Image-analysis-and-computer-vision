import torch
import torch.nn as nn
import time
from pathlib import Path


def freeze_all_layers(model):
    """Freeze all parameters in the model."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_layers(module):
    """Unfreeze parameters in a module (or model.fc, model.layer4, etc)."""
    for p in module.parameters():
        p.requires_grad = True


def count_trainable(model):
    """Utility: count parameters that will actually update."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def step(self, current):
        if self.best is None or current < self.best - self.min_delta:
            self.best = current
            self.counter = 0
            return False  # do not stop
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    optimizer,
    device,
    scheduler=None,
    save_path="best_model.pt",
    early_stopping=True,
    patience=7,
):
    """
    Train/validate with early stopping + scheduler + checkpoint saving.
    Returns training history.
    """

    criterion = nn.CrossEntropyLoss()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    stopper = EarlyStopper(patience=patience) if early_stopping else None

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = -1

    for epoch in range(num_epochs):
        t0 = time.time()

        ### ---------------- TRAIN ---------------- ###
        model.train()
        running_loss = 0
        running_corrects = 0
        n = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            n += images.size(0)

        train_loss = running_loss / n
        train_acc = running_corrects / n

        ### ---------------- VALIDATION ---------------- ###
        model.eval()
        val_loss = 0
        val_corrects = 0
        vn = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels).item()
                vn += images.size(0)

        val_loss /= vn
        val_acc = val_corrects / vn

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Step LR scheduler (if provided)
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        # Print log
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {time.time()-t0:.1f}s"
        )

        # Early stopping
        if early_stopping:
            if stopper.step(val_loss):
                print("Early stopping triggered.")
                break

    print("Training finished. Best val acc:", best_val_acc)
    return history
