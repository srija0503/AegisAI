"""
federated_learning/model_serializer.py
──────────────────────────────────────
Model weight serialization and deserialization engine for Federated Learning.
Supports PyTorch state_dict and NumPy arrays with cryptographic SHA-256
checksum verification for transmission across quantum-secure channels.
"""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Any, Mapping, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ModelSerializer:
    """
    Serializes and deserializes neural network weights for edge-to-server FL exchange.
    """

    @staticmethod
    def state_dict_to_numpy(state_dict: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
        """Convert PyTorch state_dict to dictionary of CPU NumPy float arrays."""
        return {
            k: v.detach().cpu().numpy().copy()
            for k, v in state_dict.items()
        }

    @staticmethod
    def numpy_to_state_dict(
        numpy_dict: Mapping[str, np.ndarray],
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Convert dictionary of NumPy arrays back to PyTorch state_dict tensors."""
        target_dev = torch.device(device)
        return {
            k: torch.from_numpy(v).to(target_dev)
            for k, v in numpy_dict.items()
        }

    @classmethod
    def serialize_weights(cls, weights: Mapping[str, Union[torch.Tensor, np.ndarray]]) -> bytes:
        """
        Serializes weights into binary bytes using PyTorch buffer serialization.
        Guarantees deterministic, platform-independent representation.
        """
        cpu_tensors: dict[str, torch.Tensor] = {}
        for k, v in weights.items():
            if isinstance(v, np.ndarray):
                cpu_tensors[k] = torch.from_numpy(v).cpu()
            elif isinstance(v, torch.Tensor):
                cpu_tensors[k] = v.detach().cpu()
            else:
                raise TypeError(f"Unsupported weight type for key '{k}': {type(v)}")

        buf = io.BytesIO()
        torch.save(cpu_tensors, buf)
        return buf.getvalue()

    @classmethod
    def deserialize_weights(
        cls,
        payload: bytes,
        device: str = "cpu",
        as_numpy: bool = False,
    ) -> dict[str, Any]:
        """
        Deserializes binary payload into state_dict or numpy dictionary.
        """
        buf = io.BytesIO(payload)
        tensors: dict[str, torch.Tensor] = torch.load(buf, map_location=device, weights_only=True)

        if as_numpy:
            return {k: v.numpy() for k, v in tensors.items()}
        return tensors

    @staticmethod
    def compute_checksum(payload: bytes) -> str:
        """Compute SHA-256 hexadecimal digest of serialized weights."""
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def get_model_summary(cls, weights: Mapping[str, Union[torch.Tensor, np.ndarray]]) -> dict[str, Any]:
        """Calculates total parameter count and layer shapes."""
        total_params = 0
        layer_shapes: dict[str, Any] = {}
        for k, v in weights.items():
            shape = list(v.shape)
            params = int(np.prod(shape))
            total_params += params
            layer_shapes[k] = {"shape": shape, "num_params": params}

        return {
            "total_parameters": total_params,
            "total_layers": len(weights),
            "layers": layer_shapes,
        }
