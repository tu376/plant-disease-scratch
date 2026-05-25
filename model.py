from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable, Iterator, Sequence


class Layer:
    """Base class for all neural network layers."""

    def __init__(self) -> None:
        self.training = True

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    def forward(self, x: Any) -> Any:
        """Compute output of this layer."""
        raise NotImplementedError("forward() must be implemented by subclass")

    def backward(self, dout: Any) -> Any:
        """Propagate gradient to the previous layer."""
        raise NotImplementedError("backward() must be implemented by subclass")

    def parameters(self) -> list[dict[str, Any]]:
        """Return trainable parameters of this layer.

        Example for Linear:
            [
                {"name": "W", "param": self.W, "grad": self.dW},
                {"name": "b", "param": self.b, "grad": self.db},
            ]
        """
        return []

    def zero_grad(self) -> None:
        """Set all parameter gradients to zero."""
        for item in self.parameters():
            grad = item.get("grad")
            if grad is not None and hasattr(grad, "fill"):
                grad.fill(0)

    def train(self) -> "Layer":
        """Switch layer to training mode."""
        self.training = True
        return self

    def eval(self) -> "Layer":
        """Switch layer to evaluation mode."""
        self.training = False
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return trainable parameters for saving."""
        state: dict[str, Any] = {}

        for item in self.parameters():
            name = item.get("name")
            param = item.get("param")

            if name is not None and param is not None:
                state[str(name)] = param.copy() if hasattr(param, "copy") else param

        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load trainable parameters from dictionary."""
        for item in self.parameters():
            name = item.get("name")
            param = item.get("param")

            if name in state and param is not None:
                if hasattr(param, "__setitem__"):
                    param[...] = state[name]
                else:
                    item["param"] = state[name]


class Sequential(Layer):
    """A container that runs layers in order."""

    def __init__(self, layers: Sequence[Layer] | None = None) -> None:
        super().__init__()
        self.layers: list[Layer] = list(layers or [])

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def __getitem__(self, index: int) -> Layer:
        return self.layers[index]

    def add(self, layer: Layer) -> None:
        """Add one layer to the model."""
        if not isinstance(layer, Layer):
            raise TypeError("layer must inherit from Layer")

        self.layers.append(layer)

    def forward(self, x: Any) -> Any:
        """Run forward pass through all layers."""
        out = x

        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, dout: Any) -> Any:
        """Run backward pass through all layers in reverse order."""
        grad = dout

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def parameters(self) -> list[dict[str, Any]]:
        """Return all trainable parameters from all layers."""
        params: list[dict[str, Any]] = []

        for layer_index, layer in enumerate(self.layers):
            for item in layer.parameters():
                record = dict(item)
                name = record.get("name", "param")
                record["name"] = f"{layer_index}.{name}"
                params.append(record)

        return params

    def named_parameters(self) -> Iterable[tuple[str, Any, Any]]:
        """Yield (name, parameter, gradient)."""
        for item in self.parameters():
            yield item["name"], item.get("param"), item.get("grad")

    def zero_grad(self) -> None:
        """Clear gradients of all layers."""
        for layer in self.layers:
            layer.zero_grad()

    def train(self) -> "Sequential":
        """Switch model and all layers to training mode."""
        self.training = True

        for layer in self.layers:
            layer.train()

        return self

    def eval(self) -> "Sequential":
        """Switch model and all layers to evaluation mode."""
        self.training = False

        for layer in self.layers:
            layer.eval()

        return self

    def state_dict(self) -> dict[str, Any]:
        """Return model parameters for saving checkpoint."""
        state: dict[str, Any] = OrderedDict()

        for item in self.parameters():
            name = item.get("name")
            param = item.get("param")

            if name is not None and param is not None:
                state[name] = param.copy() if hasattr(param, "copy") else param

        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load model parameters from checkpoint."""
        for item in self.parameters():
            name = item.get("name")
            param = item.get("param")

            if name in state and param is not None:
                if hasattr(param, "__setitem__"):
                    param[...] = state[name]
                else:
                    item["param"] = state[name]

    def summary(self) -> str:
        """Return readable model structure."""
        lines = ["Sequential("]

        for index, layer in enumerate(self.layers):
            lines.append(f"  ({index}): {layer.__class__.__name__}")

        lines.append(")")
        return "\n".join(lines)


class Flatten(Layer):
    """Flatten all dimensions except batch dimension."""

    def __init__(self) -> None:
        super().__init__()
        self.input_shape: tuple[int, ...] | None = None

    def forward(self, x: Any) -> Any:
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: Any) -> Any:
        if self.input_shape is None:
            raise RuntimeError("forward() must be called before backward()")

        return dout.reshape(self.input_shape)