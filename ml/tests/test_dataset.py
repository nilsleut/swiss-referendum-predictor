import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from src.dataset import ReferendumDataset, create_dataloaders


@pytest.fixture
def small_arrays():
    rng = np.random.default_rng(0)
    X = rng.random((50, 10)).astype(np.float32)
    y = rng.random(50).astype(np.float32) * 100
    return X, y


def test_len(small_arrays) -> None:
    X, y = small_arrays
    ds = ReferendumDataset(X, y)
    assert len(ds) == 50


def test_feature_shape(small_arrays) -> None:
    X, y = small_arrays
    ds = ReferendumDataset(X, y)
    feat, _ = ds[0]
    assert feat.shape == (10,)


def test_target_shape(small_arrays) -> None:
    X, y = small_arrays
    ds = ReferendumDataset(X, y)
    _, target = ds[0]
    assert target.shape == (1,)


def test_dtypes(small_arrays) -> None:
    X, y = small_arrays
    ds = ReferendumDataset(X, y)
    feat, target = ds[0]
    assert feat.dtype == torch.float32
    assert target.dtype == torch.float32


def test_values_preserved() -> None:
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    y = np.array([42.0], dtype=np.float32)
    ds = ReferendumDataset(X, y)
    feat, target = ds[0]
    assert torch.allclose(feat, torch.tensor([1.0, 2.0]))
    assert torch.allclose(target, torch.tensor([42.0]))


def test_dataloader_batch_shape(small_arrays) -> None:
    X, y = small_arrays
    ds = ReferendumDataset(X, y)
    loader = DataLoader(ds, batch_size=8)
    feat_batch, target_batch = next(iter(loader))
    assert feat_batch.shape == (8, 10)
    assert target_batch.shape == (8, 1)


def test_create_dataloaders_sizes(small_arrays) -> None:
    X, y = small_arrays
    train_ds = ReferendumDataset(X[:35], y[:35])
    val_ds   = ReferendumDataset(X[35:42], y[35:42])
    test_ds  = ReferendumDataset(X[42:], y[42:])
    train_l, val_l, test_l = create_dataloaders(train_ds, val_ds, test_ds, batch_size=16)
    assert len(train_l.dataset) == 35  # type: ignore[arg-type]
    assert len(val_l.dataset)   == 7   # type: ignore[arg-type]
    assert len(test_l.dataset)  == 8   # type: ignore[arg-type]
