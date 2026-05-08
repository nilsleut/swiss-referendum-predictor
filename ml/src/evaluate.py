"""
Evaluate a trained MLP checkpoint on the test set.

Run:
    python -m src.evaluate --config experiments/default.yaml --checkpoint models/checkpoints/best_model.pt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml

from src.dataset import load_and_split
from src.model import MLP


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return MAE, RMSE, and R² for a set of predictions."""
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _collect_predictions(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device)).cpu().numpy().flatten()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy().flatten())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def evaluate(config: Dict, checkpoint_path: str) -> Dict[str, float]:
    """Load checkpoint, run on test set, print and return metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = checkpoint.get("config", config)
    feature_columns = checkpoint.get("feature_columns")

    data_cfg = saved_config["data"]
    _, _, test_ds, _scaler, actual_feature_columns = load_and_split(
        features_path=data_cfg["features_path"],
        targets_path=data_cfg["targets_path"],
        features_sep=data_cfg.get("features_sep", ";"),
        val_size=data_cfg.get("val_size", 0.15),
        test_size=data_cfg.get("test_size", 0.15),
        random_seed=data_cfg.get("random_seed", 42),
    )

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model_cfg = saved_config["model"]
    model = MLP(
        input_dim=len(actual_feature_columns),
        hidden_layers=model_cfg["hidden_layers"],
        dropout=model_cfg.get("dropout", 0.3),
        batch_norm=model_cfg.get("batch_norm", True),
        activation=model_cfg.get("activation", "relu"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_pred, y_true = _collect_predictions(test_loader, model, device)
    metrics = compute_metrics(y_true, y_pred)

    print("\n=== Test Set Metrics ===")
    print(f"  MAE  : {metrics['mae']:.4f}")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  R²   : {metrics['r2']:.4f}")
    print(f"  Samples: {len(y_true)}")

    metrics_path = Path(checkpoint_path).parent / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}\n")

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--config", type=str, default="experiments/default.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_model.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg, args.checkpoint)
