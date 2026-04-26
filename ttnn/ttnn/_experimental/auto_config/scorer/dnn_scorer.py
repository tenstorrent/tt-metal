# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
DNN-based configuration generator for matmul operations.

Instead of just scoring candidate configs, this DNN **generates** the optimal
config directly from input features. This addresses the maintainer requirement:
"Have the DNN generate the configuration, instead of just scoring it."

Architecture (multi-head MLP):
    Input features (M, K, N, batch, dtype, layout, grid, arch, ...)
        → Shared backbone: Linear(256) → ReLU → Dropout → Linear(128) → ReLU → Dropout
        → Head 1: config_family (7-class softmax)
        → Head 2: in0_block_w (regression, clamped to valid divisor)
        → Head 3: per_core_M (regression)
        → Head 4: per_core_N (regression)
        → Head 5: out_subblock_h (regression, constrained)
        → Head 6: out_subblock_w (regression, constrained)
        → Head 7: math_fidelity (4-class softmax)
        → Head 8: mcast_in0 (binary sigmoid)

Weights are stored using torch.save (.pt format) per maintainer requirement.

Retraining:
    python -m ttnn._experimental.auto_config.retrain_dnn --full
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Feature keys (order matters for vector construction) ---
DNN_FEATURE_KEYS = [
    "M",
    "K",
    "N",
    "batch_size_a",
    "batch_size_b",
    "M_tiles",
    "K_tiles",
    "N_tiles",
    "grid_x",
    "grid_y",
    "num_cores",
    "num_devices",
]

# --- Config families (output classes) ---
CONFIG_FAMILIES = [
    "MultiCast1D",
    "MultiCast2D",
    "Reuse",
    "DRAMSharded",
    "BatchedDRAMSharded",
    "MinimalMatmul",
    "MultiCore",
]

# --- Math fidelity classes ---
MATH_FIDELITIES = ["LoFi", "HiFi2", "HiFi3", "HiFi4"]

# Boolean/categorical input features
BOOL_FEATURE_KEYS = [
    "is_a_sharded",
    "is_b_sharded",
    "is_batched_b",
    "is_fp32_accumulate",
]

# Dtype one-hot keys
DTYPE_CLASSES = ["BFLOAT16", "BFLOAT8_B", "BFLOAT4_B", "FLOAT32", "OTHER"]


def _features_to_vector(features: Dict[str, Any]) -> List[float]:
    """Convert matmul features to a flat numeric input vector for the DNN."""
    vec = []

    # Numeric features: log-normalize
    for key in DNN_FEATURE_KEYS:
        val = features.get(key, 0)
        if isinstance(val, bool):
            vec.append(1.0 if val else 0.0)
        elif isinstance(val, (int, float)):
            vec.append(math.log2(max(1, val)))
        else:
            vec.append(0.0)

    # Boolean features
    for key in BOOL_FEATURE_KEYS:
        vec.append(1.0 if features.get(key, False) else 0.0)

    # Dtype one-hot (input A)
    dtype_a = str(features.get("dtype_a", ""))
    for cls in DTYPE_CLASSES:
        vec.append(1.0 if cls in dtype_a.upper() else 0.0)

    # Dtype one-hot (input B)
    dtype_b = str(features.get("dtype_b", ""))
    for cls in DTYPE_CLASSES:
        vec.append(1.0 if cls in dtype_b.upper() else 0.0)

    return vec


# Total input dimension
INPUT_DIM = len(DNN_FEATURE_KEYS) + len(BOOL_FEATURE_KEYS) + 2 * len(DTYPE_CLASSES)
# = 12 + 4 + 10 = 26


class DNNConfigGenerator:
    """
    DNN-based config generator using a multi-head MLP.

    Instead of scoring candidates, this model directly predicts the optimal
    matmul configuration parameters from input features.

    The model outputs:
        - config_family: which MatmulProgramConfig class to use
        - in0_block_w: inner loop block width
        - per_core_M: tiles per core in M dimension
        - per_core_N: tiles per core in N dimension
        - out_subblock_h: output subblock height
        - out_subblock_w: output subblock width
        - math_fidelity: which MathFidelity to use
        - mcast_in0: whether to multicast input 0 (for 1D configs)
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model = None
        self._model_path = model_path or self._default_model_path()
        self._load_model()

    @staticmethod
    def _default_model_path() -> str:
        return os.path.join(
            os.path.expanduser("~"),
            ".ttnn",
            "auto_config_cache",
            "dnn_config_generator.pt",
        )

    def _load_model(self) -> None:
        """Load model weights from disk using torch.load."""
        path = Path(self._model_path)
        if not path.exists():
            logger.debug("DNN config generator model not found at %s, using fallback", path)
            self._model = None
            return

        try:
            import torch

            self._model = torch.load(path, map_location="cpu", weights_only=False)
            self._model.eval()
            logger.debug("Loaded DNN config generator from %s", path)
        except Exception as e:
            logger.warning("Failed to load DNN config generator: %s", e)
            self._model = None

    def is_available(self) -> bool:
        """Check if a trained model is loaded and ready."""
        return self._model is not None

    def generate(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate optimal config parameters from input features.

        Returns a dict with keys:
            config_family, in0_block_w, per_core_M, per_core_N,
            out_subblock_h, out_subblock_w, math_fidelity, mcast_in0

        Returns None if no model is loaded.
        """
        if self._model is None:
            return None

        try:
            import torch

            vec = _features_to_vector(features)
            x = torch.tensor([vec], dtype=torch.float32)

            with torch.no_grad():
                outputs = self._model(x)

            return self._decode_outputs(outputs, features)

        except Exception as e:
            logger.warning("DNN config generation failed: %s", e)
            return None

    def _decode_outputs(self, outputs: Dict, features: Dict[str, Any]) -> Dict[str, Any]:
        """Decode raw model outputs into valid config parameters."""
        import torch

        result = {}

        # Config family (argmax of softmax)
        family_logits = outputs["config_family"]
        family_idx = torch.argmax(family_logits, dim=-1).item()
        result["config_family"] = CONFIG_FAMILIES[family_idx]

        # Math fidelity (argmax of softmax)
        fidelity_logits = outputs["math_fidelity"]
        fidelity_idx = torch.argmax(fidelity_logits, dim=-1).item()
        result["math_fidelity"] = MATH_FIDELITIES[fidelity_idx]

        # mcast_in0 (sigmoid > 0.5)
        result["mcast_in0"] = outputs["mcast_in0"].item() > 0.5

        # Regression outputs — clamp to valid ranges and round
        M_tiles = max(1, features.get("M_tiles", 1))
        K_tiles = max(1, features.get("K_tiles", 1))
        N_tiles = max(1, features.get("N_tiles", 1))
        grid_x = features.get("grid_x", 8)
        grid_y = features.get("grid_y", 8)
        num_cores = grid_x * grid_y

        # in0_block_w: must divide K_tiles
        raw_in0_bw = max(1, round(outputs["in0_block_w"].item()))
        result["in0_block_w"] = self._snap_to_divisor(raw_in0_bw, K_tiles, max_val=8)

        # per_core_M
        raw_pcm = max(1, round(outputs["per_core_M"].item()))
        result["per_core_M"] = max(1, min(raw_pcm, M_tiles))

        # per_core_N
        raw_pcn = max(1, round(outputs["per_core_N"].item()))
        result["per_core_N"] = max(1, min(raw_pcn, N_tiles))

        # Subblock dims: h*w <= 8
        raw_h = max(1, min(8, round(outputs["out_subblock_h"].item())))
        raw_w = max(1, min(8, round(outputs["out_subblock_w"].item())))
        # Enforce constraint
        while raw_h * raw_w > 8:
            if raw_w > raw_h:
                raw_w -= 1
            else:
                raw_h -= 1
        # Snap to divisors of per_core dims
        result["out_subblock_h"] = self._snap_to_divisor(raw_h, result["per_core_M"], max_val=8)
        result["out_subblock_w"] = self._snap_to_divisor(raw_w, result["per_core_N"], max_val=8)

        return result

    @staticmethod
    def _snap_to_divisor(value: int, total: int, max_val: int = 8) -> int:
        """Snap a value to the nearest divisor of total, within [1, max_val]."""
        value = max(1, min(value, max_val))
        # Try exact value first
        if total % value == 0:
            return value
        # Search outward
        for delta in range(1, max_val):
            for candidate in [value - delta, value + delta]:
                if 1 <= candidate <= max_val and total % candidate == 0:
                    return candidate
        return 1

    def save_model(self, path: Optional[str] = None) -> None:
        """Save model weights using torch.save."""
        if self._model is None:
            logger.warning("No model to save")
            return
        path = path or self._model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import torch

        torch.save(self._model, path)
        logger.info("DNN config generator saved to %s", path)

    @staticmethod
    def build_model() -> "torch.nn.Module":
        """Build the multi-head MLP architecture."""
        import torch
        import torch.nn as nn

        class MultiHeadMatmulConfigNet(nn.Module):
            """Multi-head MLP that predicts optimal matmul config from features."""

            def __init__(self, input_dim: int = INPUT_DIM):
                super().__init__()
                # Shared backbone
                self.backbone = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
                # Classification heads
                self.config_family_head = nn.Linear(128, len(CONFIG_FAMILIES))
                self.math_fidelity_head = nn.Linear(128, len(MATH_FIDELITIES))
                self.mcast_in0_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
                # Regression heads (predict log2 values, decoded via 2^x rounding)
                self.in0_block_w_head = nn.Sequential(nn.Linear(128, 1), nn.ReLU())
                self.per_core_M_head = nn.Sequential(nn.Linear(128, 1), nn.ReLU())
                self.per_core_N_head = nn.Sequential(nn.Linear(128, 1), nn.ReLU())
                self.out_subblock_h_head = nn.Sequential(nn.Linear(128, 1), nn.ReLU())
                self.out_subblock_w_head = nn.Sequential(nn.Linear(128, 1), nn.ReLU())

            def forward(self, x):
                h = self.backbone(x)
                return {
                    "config_family": self.config_family_head(h),
                    "math_fidelity": self.math_fidelity_head(h),
                    "mcast_in0": self.mcast_in0_head(h).squeeze(-1),
                    "in0_block_w": self.in0_block_w_head(h).squeeze(-1),
                    "per_core_M": self.per_core_M_head(h).squeeze(-1),
                    "per_core_N": self.per_core_N_head(h).squeeze(-1),
                    "out_subblock_h": self.out_subblock_h_head(h).squeeze(-1),
                    "out_subblock_w": self.out_subblock_w_head(h).squeeze(-1),
                }

        return MultiHeadMatmulConfigNet()

    def train_model(
        self,
        training_data: List[Dict[str, Any]],
        epochs: int = 200,
        lr: float = 0.001,
        batch_size: int = 256,
    ) -> None:
        """
        Train the DNN config generator on benchmark telemetry data.

        Args:
            training_data: List of dicts with keys:
                - "features": Dict of matmul features
                - "best_config": Dict with optimal config params
                    (config_family, in0_block_w, per_core_M, per_core_N,
                     out_subblock_h, out_subblock_w, math_fidelity, mcast_in0)
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Mini-batch size
        """
        if not training_data:
            logger.warning("No training data provided")
            return

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.error("PyTorch required for training. Install with: pip install torch")
            return

        logger.info("Training DNN config generator on %d samples for %d epochs", len(training_data), epochs)

        # Build model
        model = self.build_model()

        # Prepare data tensors
        inputs = []
        family_targets = []
        fidelity_targets = []
        mcast_targets = []
        in0_bw_targets = []
        pcm_targets = []
        pcn_targets = []
        sub_h_targets = []
        sub_w_targets = []

        for data in training_data:
            vec = _features_to_vector(data["features"])
            inputs.append(vec)

            cfg = data["best_config"]
            family_targets.append(CONFIG_FAMILIES.index(cfg.get("config_family", "MultiCast1D")))
            fidelity_targets.append(MATH_FIDELITIES.index(cfg.get("math_fidelity", "HiFi4")))
            mcast_targets.append(1.0 if cfg.get("mcast_in0", False) else 0.0)
            in0_bw_targets.append(float(cfg.get("in0_block_w", 1)))
            pcm_targets.append(float(cfg.get("per_core_M", 1)))
            pcn_targets.append(float(cfg.get("per_core_N", 1)))
            sub_h_targets.append(float(cfg.get("out_subblock_h", 1)))
            sub_w_targets.append(float(cfg.get("out_subblock_w", 1)))

        X = torch.tensor(inputs, dtype=torch.float32)
        y_family = torch.tensor(family_targets, dtype=torch.long)
        y_fidelity = torch.tensor(fidelity_targets, dtype=torch.long)
        y_mcast = torch.tensor(mcast_targets, dtype=torch.float32)
        y_in0_bw = torch.tensor(in0_bw_targets, dtype=torch.float32)
        y_pcm = torch.tensor(pcm_targets, dtype=torch.float32)
        y_pcn = torch.tensor(pcn_targets, dtype=torch.float32)
        y_sub_h = torch.tensor(sub_h_targets, dtype=torch.float32)
        y_sub_w = torch.tensor(sub_w_targets, dtype=torch.float32)

        dataset = TensorDataset(X, y_family, y_fidelity, y_mcast, y_in0_bw, y_pcm, y_pcn, y_sub_h, y_sub_w)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loss functions
        ce_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                bx, bf, bfi, bm, biw, bpm, bpn, bsh, bsw = batch
                optimizer.zero_grad()

                outputs = model(bx)

                loss = (
                    ce_loss(outputs["config_family"], bf) * 2.0  # Weight family prediction higher
                    + ce_loss(outputs["math_fidelity"], bfi)
                    + bce_loss(outputs["mcast_in0"], bm)
                    + mse_loss(outputs["in0_block_w"], biw)
                    + mse_loss(outputs["per_core_M"], bpm) * 0.5
                    + mse_loss(outputs["per_core_N"], bpn) * 0.5
                    + mse_loss(outputs["out_subblock_h"], bsh)
                    + mse_loss(outputs["out_subblock_w"], bsw)
                )

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            if epoch % 20 == 0:
                logger.info(
                    "Epoch %d/%d: loss=%.4f, lr=%.6f",
                    epoch,
                    epochs,
                    total_loss / len(loader),
                    scheduler.get_last_lr()[0],
                )

        self._model = model
        self._model.eval()
        self.save_model()
        logger.info("DNN config generator training complete")


# --- Legacy scorer interface (backward compatible) ---


class DNNScorer:
    """
    Legacy DNN scorer interface. Uses the config generator internally
    but exposes a score() method for backward compatibility.

    If a trained generator model exists, it uses the generator's confidence
    for scoring. Otherwise falls back to random weight initialization.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._generator = DNNConfigGenerator(model_path=model_path)

    def score(self, candidate: "ConfigCandidate", features: Dict[str, Any]) -> float:
        """Score a candidate by checking how close it is to the DNN's prediction."""
        if not self._generator.is_available():
            return 0.5  # No model, neutral score

        predicted = self._generator.generate(features)
        if predicted is None:
            return 0.5

        # Score = similarity between candidate and DNN prediction
        score = 0.0
        total_weight = 0.0

        # Family match (highest weight)
        if candidate.config_family == predicted.get("config_family"):
            score += 3.0
        total_weight += 3.0

        # Parameter similarity
        for key in ["in0_block_w", "per_core_M", "per_core_N"]:
            pred_val = predicted.get(key, 1)
            cand_val = candidate.params.get(key, 1)
            if pred_val > 0 and cand_val > 0:
                ratio = min(pred_val, cand_val) / max(pred_val, cand_val)
                score += ratio
            total_weight += 1.0

        return score / total_weight if total_weight > 0 else 0.5

    def predict(self, features: Dict[str, Any], config_params: Dict[str, Any]) -> float:
        """Legacy predict interface."""
        return 0.5
