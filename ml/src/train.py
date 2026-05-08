"""
Training loop for Swiss referendum turnout MLP.

Run:
    python -m src.train --config experiments/default.yaml
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.dataset import create_dataloaders, load_and_split
from src.model import MLP


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when val loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.best_state: Optional[Dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Epoch-level helpers
# ---------------------------------------------------------------------------

def _run_epoch(
    loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: Dict) -> Path:
    """Train the MLP and return the path to the saved best checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # --- Data ---
    data_cfg = config["data"]
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, _scaler, feature_columns = load_and_split(
        features_path=data_cfg["features_path"],
        targets_path=data_cfg["targets_path"],
        features_sep=data_cfg.get("features_sep", ";"),
        val_size=data_cfg.get("val_size", 0.15),
        test_size=data_cfg.get("test_size", 0.15),
        random_seed=data_cfg.get("random_seed", 42),
        scaler_save_path=str(checkpoint_dir / "scaler.joblib"),
        metadata_save_path=str(checkpoint_dir / "metadata.json"),
    )

    train_cfg = config["training"]
    train_loader, val_loader, _test_loader = create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=train_cfg.get("batch_size", 32),
    )

    # --- Model ---
    model_cfg = config["model"]
    model = MLP(
        input_dim=len(feature_columns),
        hidden_layers=model_cfg["hidden_layers"],
        dropout=model_cfg.get("dropout", 0.3),
        batch_norm=model_cfg.get("batch_norm", True),
        activation=model_cfg.get("activation", "relu"),
    ).to(device)
    print(f"Model parameters: {model.num_parameters:,}")

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=train_cfg.get("lr_scheduler_patience", 5),
        factor=train_cfg.get("lr_scheduler_factor", 0.5),
        min_lr=1e-6,
    )
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=train_cfg.get("early_stopping_patience", 10))

    # --- Wandb ---
    log_cfg = config.get("logging", {})
    use_wandb = WANDB_AVAILABLE and log_cfg.get("enabled", True)
    if use_wandb:
        wandb.init(
            project=log_cfg.get("project", "swiss-referendum-predictor"),
            name=log_cfg.get("run_name", "baseline"),
            config=config,
        )
        wandb.watch(model, log_freq=100)

    # --- Training loop ---
    epochs = train_cfg.get("epochs", 200)
    log_every = log_cfg.get("log_every_n_epochs", 1)
    best_checkpoint_path = checkpoint_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(train_loader, model, criterion, optimizer, device)
        val_loss = _run_epoch(val_loader, model, criterion, None, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % log_every == 0:
            print(
                f"Epoch {epoch:4d}/{epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | lr={current_lr:.2e}"
            )

        if use_wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "lr": current_lr}, step=epoch)

        if early_stop.step(val_loss, model):
            print(f"Early stopping at epoch {epoch} (best val_loss={early_stop.best_loss:.4f})")
            break

    # Restore best weights and save
    early_stop.restore_best(model)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "feature_columns": feature_columns,
            "best_val_loss": early_stop.best_loss,
        },
        best_checkpoint_path,
    )
    print(f"Best model saved to {best_checkpoint_path}")

    if use_wandb:
        wandb.save(str(best_checkpoint_path))
        wandb.finish()

    return best_checkpoint_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swiss referendum turnout MLP")
    parser.add_argument("--config", type=str, default="experiments/default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
