"""
PyTorch Dataset, DataLoader, and train/val/test splitting for Swiss referendum data.

The dataset is stored in two headerless files:
  - grosserDatensatzEingabe.csv  — 6083 × 584 features, semicolon-delimited
  - grosserDatensatzAusgabe.csv  — 6083 × 1 turnout percentages, one value per line
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class ReferendumDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed referendum features and turnout target."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def load_and_split(
    features_path: str,
    targets_path: str,
    features_sep: str = ";",
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
    scaler_save_path: Optional[str] = None,
    metadata_save_path: Optional[str] = None,
) -> Tuple[
    "ReferendumDataset",
    "ReferendumDataset",
    "ReferendumDataset",
    StandardScaler,
    List[str],
]:
    """Load the two-file dataset, scale, and split into train/val/test.

    Features file: headerless, semicolon-delimited (grosserDatensatzEingabe.csv).
    Targets file:  headerless, one float per line (grosserDatensatzAusgabe.csv).

    Scaler is fit only on the training fold to prevent data leakage.
    Saves the scaler and feature-name list so inference can reproduce preprocessing.

    Returns:
        train_ds, val_ds, test_ds, scaler, feature_names
    """
    X = pd.read_csv(features_path, sep=features_sep, header=None).values.astype(np.float32)
    y = np.loadtxt(targets_path, dtype=np.float32)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Row count mismatch: features has {X.shape[0]} rows, targets has {y.shape[0]} rows"
        )

    # Named f0 … f<n> for metadata / inference
    feature_names = [f"f{i}" for i in range(X.shape[1])]

    # Split: 70 / 15 / 15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_seed, shuffle=True
    )
    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=random_seed
    )

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if scaler_save_path:
        Path(scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_save_path)

    if metadata_save_path:
        Path(metadata_save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_save_path, "w") as f:
            json.dump({"feature_names": feature_names, "n_features": X.shape[1]}, f, indent=2)

    print(
        f"Data loaded: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test | "
        f"{X.shape[1]} features"
    )

    return (
        ReferendumDataset(X_train, y_train),
        ReferendumDataset(X_val, y_val),
        ReferendumDataset(X_test, y_test),
        scaler,
        feature_names,
    )


def create_dataloaders(
    train_ds: ReferendumDataset,
    val_ds: ReferendumDataset,
    test_ds: ReferendumDataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap datasets in DataLoaders. Train loader shuffles; val/test do not."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
