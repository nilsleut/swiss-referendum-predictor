"""
ONNX-based inference for single referendum turnout predictions.

This module is the sole inference path used by the TypeScript API.
It loads the ONNX model and the scaler saved during training so that
any new input is preprocessed identically to the training data.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import onnxruntime as ort


@dataclass
class PredictionResult:
    predicted_turnout: float
    model_version: str
    feature_columns: List[str]


class ReferendumPredictor:
    """Loads an ONNX model + scaler once and exposes a predict() method."""

    def __init__(
        self,
        onnx_path: str,
        scaler_path: str,
        metadata_path: str,
        model_version: str = "v1.0.0",
    ) -> None:
        self.model_version = model_version

        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._scaler = joblib.load(scaler_path)

        with open(metadata_path) as f:
            meta = json.load(f)
        self.feature_columns: List[str] = meta["feature_columns"]

        self._input_name = self._session.get_inputs()[0].name

    def predict(self, features: Dict[str, float]) -> PredictionResult:
        """Predict turnout from a dict mapping feature name → value.

        Missing features default to 0.0 so partial inputs don't crash.
        """
        x = np.array(
            [features.get(col, 0.0) for col in self.feature_columns],
            dtype=np.float32,
        ).reshape(1, -1)

        x_scaled = self._scaler.transform(x).astype(np.float32)

        outputs = self._session.run(None, {self._input_name: x_scaled})
        turnout = float(outputs[0].flatten()[0])

        return PredictionResult(
            predicted_turnout=round(turnout, 2),
            model_version=self.model_version,
            feature_columns=self.feature_columns,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def load_predictor(
    model_dir: str = "models",
    model_version: str = "v1.0.0",
) -> ReferendumPredictor:
    """Load predictor from the standard directory layout produced by training."""
    base = Path(model_dir)
    return ReferendumPredictor(
        onnx_path=str(base / "onnx" / "model.onnx"),
        scaler_path=str(base / "checkpoints" / "scaler.joblib"),
        metadata_path=str(base / "checkpoints" / "metadata.json"),
        model_version=model_version,
    )


# ---------------------------------------------------------------------------
# CLI for quick smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single prediction")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument(
        "--features",
        type=str,
        help='JSON string of feature values, e.g. \'{"feature_a": 1.5}\'',
        default="{}",
    )
    args = parser.parse_args()

    predictor = load_predictor(args.model_dir)
    features = json.loads(args.features)
    result = predictor.predict(features)
    print(f"Predicted turnout: {result.predicted_turnout}%  (model {result.model_version})")
