"""
From-scratch training utilities for deep learning transfer experiments.

Provides the standard training loop components — train_epoch, evaluate,
and fine_tune — designed to integrate with Zeno's transfer components:
  - TransferScheduler for progressive unfreezing
  - EWCLoss for Bayesian-inspired regularization
  - GPUCarbonTracker for emissions measurement

All loops are raw PyTorch with no dependency on HuggingFace Trainer.
"""

import torch
import torch.nn as nn


def train_epoch(model, dataloader, criterion, optimizer, device="cpu"):
    """
    Train for a single epoch.

    Args:
        model: nn.Module
        dataloader: training DataLoader yielding (inputs, targets)
        criterion: loss function
        optimizer: torch.optim optimizer
        device: computation device

    Returns:
        avg_loss: float, mean loss over all batches
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        inputs, targets = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, criterion, device="cpu"):
    """
    Evaluate model on a dataset.

    Args:
        model: nn.Module
        dataloader: evaluation DataLoader yielding (inputs, targets)
        criterion: loss function
        device: computation device

    Returns:
        dict with:
          - loss: float, mean loss
          - accuracy: float, classification accuracy (if applicable)
          - n_samples: int, total samples evaluated
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    n_samples = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            n_batches += 1
            n_samples += inputs.size(0)

            # Compute accuracy for classification
            if outputs.dim() >= 2 and outputs.size(-1) > 1:
                preds = outputs.argmax(dim=-1)
                correct += (preds == targets).sum().item()
            else:
                # Binary classification with single output
                preds = (outputs.squeeze() > 0).long()
                if targets.dtype in (torch.long, torch.int):
                    correct += (preds == targets).sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = correct / max(n_samples, 1)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "n_samples": n_samples,
    }


def fine_tune(model, train_loader, val_loader, epochs, optimizer, criterion,
              scheduler=None, ewc_loss=None, carbon_tracker=None,
              device="cpu", verbose=True):
    """
    Full fine-tuning loop with integrated transfer learning components.

    Combines training, evaluation, progressive unfreezing, EWC
    regularization, and carbon tracking into a single loop.

    Args:
        model: nn.Module to fine-tune
        train_loader: training DataLoader
        val_loader: validation DataLoader
        epochs: number of training epochs
        optimizer: torch.optim optimizer
        criterion: loss function
        scheduler: optional TransferScheduler for progressive unfreezing
        ewc_loss: optional EWCLoss for Bayesian transfer regularization
        carbon_tracker: optional GPUCarbonTracker for emissions measurement
        device: computation device
        verbose: if True, print per-epoch progress

    Returns:
        history: dict with lists of per-epoch metrics:
          - train_loss, val_loss, val_accuracy
          - co2_result (if carbon_tracker provided)
    """
    model.to(device)
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    if carbon_tracker is not None:
        carbon_tracker.start()

    for epoch in range(epochs):
        # Progressive unfreezing
        if scheduler is not None:
            scheduler.step(epoch)

        # Training
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Add EWC penalty if provided
            if ewc_loss is not None:
                loss = loss + ewc_loss(model)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # Validation
        val_result = evaluate(model, val_loader, criterion, device=device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_result["loss"])
        history["val_accuracy"].append(val_result["accuracy"])

        if verbose:
            print(f"  epoch {epoch+1:>{len(str(epochs))}}/{epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_result['loss']:.4f}  "
                  f"val_acc={val_result['accuracy']:.4f}")

    if carbon_tracker is not None:
        history["co2_result"] = carbon_tracker.stop()

    return history
