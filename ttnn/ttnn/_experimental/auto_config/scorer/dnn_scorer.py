# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Lightweight DNN-based config generator for matmul auto-config.

Uses a simple MLP to predict optimal config parameters from input features.
Falls back gracefully when no trained model file is available.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dnn_config_model.pt")


class DNNConfigGenerator:
    """MLP-based config parameter predictor.

    Predicts optimal (config_family, in0_block_w, per_core_M, per_core_N)
    from input features (M, K, N, grid_x, grid_y, dtype, etc.).
    Falls back to None when no .pt model is available.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._model = None
        self._available = False
        self._load_model()

    def _load_model(self):
        """Try to load the trained PyTorch model."""
        if not os.path.isfile(self._model_path):
            logger.debug("DNN model not found at %s — using heuristic only", self._model_path)
            return
        try:
            import torch

            self._model = torch.jit.load(self._model_path, map_location="cpu")
            self._model.eval()
            self._available = True
            logger.info("DNN config generator loaded from %s", self._model_path)
        except Exception as e:
            logger.debug("Failed to load DNN model: %s", e)
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def generate(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Predict optimal config parameters from input features.

        Returns a dict with keys: config_family, in0_block_w, per_core_M,
        per_core_N, out_subblock_h, out_subblock_w, or None if unavailable.
        """
        if not self._available or self._model is None:
            return None
        try:
            import torch

            feature_vec = torch.tensor(
                [
                    float(features.get("M", 0)),
                    float(features.get("K", 0)),
                    float(features.get("N", 0)),
                    float(features.get("grid_x", 8)),
                    float(features.get("grid_y", 8)),
                    float(features.get("batch_size_a", 1)),
                ],
                dtype=torch.float32,
            ).unsqueeze(0)

            with torch.no_grad():
                output = self._model(feature_vec)

            pred = output.squeeze(0).tolist()
            family_idx = int(round(pred[0]))
            families = ["MultiCast1D", "MultiCast2D", "DRAM"]
            return {
                "config_family": families[family_idx % len(families)],
                "in0_block_w": max(1, int(round(pred[1]))),
                "per_core_M": max(1, int(round(pred[2]))),
                "per_core_N": max(1, int(round(pred[3]))),
            }
        except Exception as e:
            logger.debug("DNN generation failed: %s", e)
            return None

    @staticmethod
    def train_from_csv(csv_path: str, output_path: str, epochs: int = 200):
        """Train a new DNN model from benchmark CSV data.

        CSV columns expected: M, K, N, grid_x, grid_y, batch_size,
        config_family, in0_block_w, per_core_M, per_core_N, latency_us
        """
        import torch
        import torch.nn as nn

        # Define simple 3-layer MLP
        class ConfigMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(6, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 4),
                )

            def forward(self, x):
                return self.net(x)

        # Load CSV
        import csv

        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            raise ValueError(f"No data in {csv_path}")

        family_map = {"MultiCast1D": 0, "MultiCast2D": 1, "DRAM": 2}
        X, Y = [], []
        for r in rows:
            X.append(
                [
                    float(r["M"]),
                    float(r["K"]),
                    float(r["N"]),
                    float(r["grid_x"]),
                    float(r["grid_y"]),
                    float(r["batch_size"]),
                ]
            )
            Y.append(
                [
                    float(family_map.get(r["config_family"], 0)),
                    float(r["in0_block_w"]),
                    float(r["per_core_M"]),
                    float(r["per_core_N"]),
                ]
            )

        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)

        model = ConfigMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            pred = model(X_t)
            loss = loss_fn(pred, Y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 50 == 0:
                logger.info("Epoch %d/%d  loss=%.4f", epoch + 1, epochs, loss.item())

        scripted = torch.jit.script(model)
        scripted.save(output_path)
        logger.info("Model saved to %s (%d training samples)", output_path, len(rows))
