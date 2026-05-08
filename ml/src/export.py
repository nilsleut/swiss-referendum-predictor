"""
Export a trained PyTorch checkpoint to ONNX format.

Also writes scaler_params.json and test_metrics.json into the ONNX directory
so the TypeScript API has everything it needs for inference without touching
the Python checkpoint files.

Run:
    python -m src.export --checkpoint models/checkpoints/best_model.pt
"""

import argparse
import json
import shutil
from pathlib import Path

import joblib
import torch

from src.model import MLP


def export_onnx(checkpoint_path: str, onnx_dir: str = "models/onnx") -> str:
    """Load a .pt checkpoint and export the model to ONNX.

    Dynamic batch size is enabled so the API can send variable-size batches.
    Alongside model.onnx, writes:
      scaler_params.json  — mean/scale arrays for the TypeScript scaler
      test_metrics.json   — MAE/RMSE/R² (copied from checkpoint dir if present)
      model_info.json     — n_features, model_version, architecture

    Returns:
        Path to the exported ONNX file.
    """
    checkpoint_path = Path(checkpoint_path)
    device = torch.device("cpu")
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    config = checkpoint["config"]
    feature_columns = checkpoint["feature_columns"]
    model_cfg = config["model"]
    n_features = len(feature_columns)

    model = MLP(
        input_dim=n_features,
        hidden_layers=model_cfg["hidden_layers"],
        dropout=model_cfg.get("dropout", 0.3),
        batch_norm=model_cfg.get("batch_norm", True),
        activation=model_cfg.get("activation", "relu"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    onnx_path = Path(onnx_dir)
    onnx_path.mkdir(parents=True, exist_ok=True)
    out_file = str(onnx_path / "model.onnx")

    dummy_input = torch.zeros(1, n_features)
    torch.onnx.export(
        model,
        dummy_input,
        out_file,
        input_names=["features"],
        output_names=["turnout"],
        dynamic_axes={"features": {0: "batch_size"}, "turnout": {0: "batch_size"}},
        opset_version=18,
        dynamo=False,
    )
    print(f"ONNX model exported  -> {out_file}")

    # --- Scaler params (needed by TypeScript API to normalise inputs) ---
    scaler_file = checkpoint_path.parent / "scaler.joblib"
    if scaler_file.exists():
        scaler = joblib.load(str(scaler_file))
        scaler_params = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "n_features": n_features,
        }
        sp_out = onnx_path / "scaler_params.json"
        with open(sp_out, "w") as f:
            json.dump(scaler_params, f)
        print(f"Scaler params written -> {sp_out}")
    else:
        print("Warning: scaler.joblib not found — skipping scaler_params.json")

    # --- Test metrics ---
    metrics_src = checkpoint_path.parent / "test_metrics.json"
    if metrics_src.exists():
        metrics_dst = onnx_path / "test_metrics.json"
        shutil.copy(metrics_src, metrics_dst)
        print(f"Test metrics copied  -> {metrics_dst}")

    # --- Model info (architecture + version for the DB seed on startup) ---
    model_info = {
        "model_version": "v1.0.0",
        "n_features": n_features,
        "hidden_layers": model_cfg["hidden_layers"],
        "dropout": model_cfg.get("dropout", 0.3),
        "batch_norm": model_cfg.get("batch_norm", True),
        "activation": model_cfg.get("activation", "relu"),
        "best_val_loss": checkpoint.get("best_val_loss"),
    }
    info_out = onnx_path / "model_info.json"
    with open(info_out, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"Model info written   -> {info_out}")

    return out_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pt")
    parser.add_argument("--onnx-dir", type=str, default="models/onnx")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export_onnx(args.checkpoint, args.onnx_dir)
