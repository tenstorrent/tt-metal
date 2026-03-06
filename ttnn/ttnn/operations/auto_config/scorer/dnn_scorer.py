# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Optional DNN-based performance scorer for matmul configurations.

Uses a small MLP to predict throughput from normalized feature vectors.
The model maps (feature_vector → predicted_throughput_score).

Retraining workflow:
    python -m ttnn.operations.auto_config.benchmark --op matmul --shapes shapes.json
    # This collects telemetry and retrains the DNN scorer.

The DNN scorer is optional — the heuristic scorer is always available as fallback.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ttnn.operations.auto_config.base import ConfigCandidate

logger = logging.getLogger(__name__)


# Feature keys used for DNN input (order matters!)
DNN_FEATURE_KEYS = [
    "M", "K", "N",
    "batch_size_a", "batch_size_b",
    "M_tiles", "K_tiles", "N_tiles",
    "grid_x", "grid_y", "num_cores",
    "num_devices",
]

# Config param keys
DNN_CONFIG_KEYS = [
    "in0_block_w", "per_core_M", "per_core_N",
    "out_subblock_h", "out_subblock_w",
]


def _features_to_vector(features: Dict[str, Any], config_params: Dict[str, Any]) -> List[float]:
    """Convert features and config params to a flat numeric vector."""
    vec = []

    # Features: log-normalize large values
    import math
    for key in DNN_FEATURE_KEYS:
        val = features.get(key, 0)
        if isinstance(val, bool):
            vec.append(1.0 if val else 0.0)
        elif isinstance(val, (int, float)):
            # Log-normalize to prevent large values from dominating
            vec.append(math.log2(max(1, val)))
        else:
            vec.append(0.0)

    # Config params
    for key in DNN_CONFIG_KEYS:
        val = config_params.get(key, 0)
        if isinstance(val, (int, float)):
            vec.append(math.log2(max(1, val)))
        else:
            vec.append(0.0)

    # Config family one-hot encoding
    families = ["MultiCast1D", "MultiCast2D", "Reuse", "DRAMSharded",
                "BatchedDRAMSharded", "MinimalMatmul", "MultiCore"]
    family = config_params.get("_family", "")
    for f in families:
        vec.append(1.0 if f == family else 0.0)

    return vec


class DNNScorer:
    """
    DNN-based performance scorer using a simple MLP.

    The model is stored as a serialized set of weights (JSON format for
    portability). For production use, this can be converted to TorchScript.

    Architecture:
        Input (n_features) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1) → Sigmoid

    Training data:
        Collected via the benchmark CLI, stored as (features, config_params, latency_us) tuples.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._weights = None
        self._model_path = model_path or self._default_model_path()
        self._load_model()

    @staticmethod
    def _default_model_path() -> str:
        return os.path.join(
            os.path.expanduser("~"), ".ttnn", "auto_config_cache", "dnn_scorer_weights.json"
        )

    def _load_model(self) -> None:
        """Load model weights from disk."""
        path = Path(self._model_path)
        if not path.exists():
            logger.debug("DNN scorer model not found, using random initialization")
            self._init_random_weights()
            return

        try:
            with open(path, "r") as f:
                self._weights = json.load(f)
            logger.debug(f"Loaded DNN scorer from {path}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load DNN scorer: {e}")
            self._init_random_weights()

    def _init_random_weights(self) -> None:
        """Initialize with small random weights (untrained model)."""
        import random
        random.seed(42)

        input_dim = len(DNN_FEATURE_KEYS) + len(DNN_CONFIG_KEYS) + 7  # +7 for family one-hot
        hidden1 = 64
        hidden2 = 32

        self._weights = {
            "w1": [[random.gauss(0, 0.1) for _ in range(input_dim)] for _ in range(hidden1)],
            "b1": [0.0] * hidden1,
            "w2": [[random.gauss(0, 0.1) for _ in range(hidden1)] for _ in range(hidden2)],
            "b2": [0.0] * hidden2,
            "w3": [[random.gauss(0, 0.1) for _ in range(hidden2)]],
            "b3": [0.0],
        }

    def _relu(self, x: List[float]) -> List[float]:
        return [max(0.0, v) for v in x]

    def _sigmoid(self, x: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, x))))

    def _matmul_vec(self, weights: List[List[float]], vec: List[float], bias: List[float]) -> List[float]:
        """Simple matrix-vector multiplication."""
        result = []
        for i, row in enumerate(weights):
            val = bias[i] + sum(w * v for w, v in zip(row, vec))
            result.append(val)
        return result

    def predict(self, features: Dict[str, Any], config_params: Dict[str, Any]) -> float:
        """
        Predict the performance score for a config.

        Returns a value in [0, 1] where higher = better predicted performance.
        """
        if self._weights is None:
            return 0.5  # No model available

        config_params_with_family = {**config_params}
        vec = _features_to_vector(features, config_params_with_family)

        # Forward pass: Input → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1) → Sigmoid
        h1 = self._matmul_vec(self._weights["w1"], vec, self._weights["b1"])
        h1 = self._relu(h1)

        h2 = self._matmul_vec(self._weights["w2"], h1, self._weights["b2"])
        h2 = self._relu(h2)

        out = self._matmul_vec(self._weights["w3"], h2, self._weights["b3"])
        return self._sigmoid(out[0])

    def score(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score a candidate using the DNN model."""
        params = {**candidate.params, "_family": candidate.config_family}
        return self.predict(features, params)

    def save_model(self, path: Optional[str] = None) -> None:
        """Save model weights to disk."""
        path = path or self._model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._weights, f)
        logger.info(f"DNN scorer saved to {path}")

    def train(self, training_data: List[Dict[str, Any]], epochs: int = 100, lr: float = 0.01) -> None:
        """
        Train the DNN scorer on benchmark telemetry data.

        Args:
            training_data: List of dicts with keys:
                - "features": Dict of matmul features
                - "config_params": Dict of config parameters
                - "latency_us": Measured latency in microseconds
            epochs: Number of training epochs
            lr: Learning rate

        Note: This is a simplified training loop. For production, consider
        using PyTorch directly for GPU-accelerated training.
        """
        if not training_data:
            logger.warning("No training data provided")
            return

        # Normalize latencies to [0, 1] scores (lower latency = higher score)
        latencies = [d["latency_us"] for d in training_data]
        min_lat = min(latencies)
        max_lat = max(latencies)
        lat_range = max(max_lat - min_lat, 1e-6)

        for data in training_data:
            data["target_score"] = 1.0 - (data["latency_us"] - min_lat) / lat_range

        logger.info(
            f"Training DNN scorer on {len(training_data)} samples for {epochs} epochs"
        )

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            logger.warning("PyTorch not found, skipping DNN scorer training")
            return

        # Build PyTorch model from current weights
        input_dim = len(self._weights["w1"][0])
        h1_dim = len(self._weights["w1"])
        h2_dim = len(self._weights["w2"])

        model = nn.Sequential(
            nn.Linear(input_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, 1),
            nn.Sigmoid()
        )

        with torch.no_grad():
            model[0].weight.copy_(torch.tensor(self._weights["w1"]))
            model[0].bias.copy_(torch.tensor(self._weights["b1"]))
            model[2].weight.copy_(torch.tensor(self._weights["w2"]))
            model[2].bias.copy_(torch.tensor(self._weights["b2"]))
            model[4].weight.copy_(torch.tensor(self._weights["w3"]))
            model[4].bias.copy_(torch.tensor(self._weights["b3"]))

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        inputs = []
        targets = []
        for data in training_data:
            params = {**data["config_params"], "_family": data.get("config_family", "")}
            vec = _features_to_vector(data["features"], params)
            inputs.append(vec)
            targets.append([data["target_score"]])

        X = torch.tensor(inputs, dtype=torch.float32)
        Y = torch.tensor(targets, dtype=torch.float32)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}: avg_loss={loss.item():.6f}")

        # Extract weights back to dict
        with torch.no_grad():
            self._weights["w1"] = model[0].weight.tolist()
            self._weights["b1"] = model[0].bias.tolist()
            self._weights["w2"] = model[2].weight.tolist()
            self._weights["b2"] = model[2].bias.tolist()
            self._weights["w3"] = model[4].weight.tolist()
            self._weights["b3"] = model[4].bias.tolist()

        self.save_model()
        logger.info("DNN scorer training complete")
