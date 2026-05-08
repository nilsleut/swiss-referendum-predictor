import pytest
import torch
from src.model import MLP


@pytest.fixture
def small_model() -> MLP:
    return MLP(input_dim=10, hidden_layers=[8, 4])


def test_forward_shape_single(small_model: MLP) -> None:
    out = small_model(torch.zeros(1, 10))
    assert out.shape == (1, 1)


def test_forward_shape_batch(small_model: MLP) -> None:
    out = small_model(torch.zeros(32, 10))
    assert out.shape == (32, 1)


def test_output_is_finite(small_model: MLP) -> None:
    out = small_model(torch.randn(16, 10))
    assert torch.isfinite(out).all()


def test_num_parameters_positive(small_model: MLP) -> None:
    assert small_model.num_parameters > 0


def test_no_batch_norm() -> None:
    model = MLP(input_dim=4, hidden_layers=[4], batch_norm=False)
    out = model(torch.zeros(2, 4))
    assert out.shape == (2, 1)


def test_zero_dropout() -> None:
    model = MLP(input_dim=4, hidden_layers=[4], dropout=0.0)
    out = model(torch.zeros(2, 4))
    assert out.shape == (2, 1)


def test_activations() -> None:
    for act in ("relu", "leaky_relu", "selu", "gelu"):
        model = MLP(input_dim=4, hidden_layers=[4], activation=act)
        out = model(torch.zeros(1, 4))
        assert out.shape == (1, 1), f"Failed for activation={act}"


def test_invalid_activation_raises() -> None:
    with pytest.raises(ValueError):
        MLP(input_dim=4, hidden_layers=[4], activation="tanh")


def test_eval_mode_deterministic(small_model: MLP) -> None:
    small_model.eval()
    x = torch.randn(4, 10)
    with torch.no_grad():
        out1 = small_model(x)
        out2 = small_model(x)
    assert torch.allclose(out1, out2)
