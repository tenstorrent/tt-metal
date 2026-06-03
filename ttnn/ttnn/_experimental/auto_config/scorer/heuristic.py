# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Heuristic performance scorer for matmul config candidates."""

from __future__ import annotations

import logging
from typing import Any, Dict

from ttnn._experimental.auto_config.base import ConfigCandidate
from ttnn._experimental.auto_config.math_fidelity import CYCLES_PER_TILE, MAX_CYCLES_PER_TILE, MathFidelity

logger = logging.getLogger(__name__)

TILE_SIZE = 32


class HeuristicScorer:
    """Rule-based performance scorer for matmul configurations."""

    def __init__(self):
        self.w_utilization = 0.25
        self.w_block_efficiency = 0.18
        self.w_layout_alignment = 0.15
        self.w_subblock_efficiency = 0.07
        self.w_backend_preference = 0.08
        self.w_fidelity_cost = 0.12
        self.w_production_bonus = 0.15

    def score(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score a candidate configuration. Returns value in [0, 1]."""
        scores = {
            "utilization": self._score_utilization(candidate, features),
            "block_efficiency": self._score_block_efficiency(candidate, features),
            "layout_alignment": self._score_layout_alignment(candidate, features),
            "subblock_efficiency": self._score_subblock_efficiency(candidate, features),
            "backend_preference": self._score_backend_preference(candidate, features),
            "fidelity_cost": self._score_fidelity_cost(candidate, features),
            "production_bonus": self._score_production_bonus(candidate, features),
        }

        total = (
            self.w_utilization * scores["utilization"]
            + self.w_block_efficiency * scores["block_efficiency"]
            + self.w_layout_alignment * scores["layout_alignment"]
            + self.w_subblock_efficiency * scores["subblock_efficiency"]
            + self.w_backend_preference * scores["backend_preference"]
            + self.w_fidelity_cost * scores["fidelity_cost"]
            + self.w_production_bonus * scores["production_bonus"]
        )

        logger.debug(
            f"Scored {candidate.config_family}: {total:.3f} "
            f"(util={scores['utilization']:.2f}, block={scores['block_efficiency']:.2f}, "
            f"layout={scores['layout_alignment']:.2f}, subblk={scores['subblock_efficiency']:.2f}, "
            f"backend={scores['backend_preference']:.2f}, fidelity={scores['fidelity_cost']:.2f}, "
            f"prod={scores['production_bonus']:.2f})"
        )

        return total

    def _score_utilization(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score based on how many cores are effectively used."""
        num_cores = features["num_cores"]
        M_tiles = features["M_tiles"]
        N_tiles = features["N_tiles"]
        batch_size = features["batch_size_a"]
        total_output_tiles = batch_size * M_tiles * N_tiles

        if candidate.config_family in ("DRAMSharded", "BatchedDRAMSharded", "MultiCore"):
            return 0.7

        config = candidate.config
        per_core_M = getattr(config, "per_core_M", M_tiles)
        per_core_N = getattr(config, "per_core_N", N_tiles)

        if per_core_M <= 0 or per_core_N <= 0:
            return 0.1

        tiles_per_core = per_core_M * per_core_N
        if tiles_per_core <= 0:
            return 0.1

        estimated_cores = max(1, total_output_tiles / tiles_per_core)
        utilization = min(1.0, estimated_cores / num_cores)
        return utilization

    def _score_block_efficiency(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score based on block sizes — larger blocks mean fewer iterations."""
        K_tiles = features["K_tiles"]
        in0_block_w = candidate.params.get("in0_block_w", 1)
        per_core_M = candidate.params.get("per_core_M", 1)
        per_core_N = candidate.params.get("per_core_N", 1)

        k_efficiency = min(1.0, in0_block_w / max(1, K_tiles))
        output_block_size = per_core_M * per_core_N
        max_block = features["M_tiles"] * features["N_tiles"]
        block_ratio = min(1.0, output_block_size / max(1, max_block)) if max_block > 0 else 0.5

        return 0.6 * k_efficiency + 0.4 * block_ratio

    def _score_layout_alignment(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score based on how well the config matches the input memory layout."""
        is_a_sharded = features["is_a_sharded"]
        is_tall = features["is_tall"]
        is_wide = features["is_wide"]
        config_family = candidate.config_family
        mcast_in0 = candidate.params.get("mcast_in0", None)

        if config_family == "MultiCast1D":
            if mcast_in0 and is_wide:
                return 1.0
            if not mcast_in0 and is_tall:
                return 1.0
            return 0.5

        if config_family == "MultiCast2D":
            if not is_tall and not is_wide:
                return 1.0
            return 0.7

        if config_family == "DRAMSharded":
            return 0.8 if not is_a_sharded else 0.3

        if config_family == "Reuse":
            return 1.0 if features["is_batched_b"] else 0.3

        return 0.5

    def _score_subblock_efficiency(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score based on subblock sizes — larger subblocks use registers better."""
        out_subblock_h = candidate.params.get("out_subblock_h", 1)
        out_subblock_w = candidate.params.get("out_subblock_w", 1)
        return min(1.0, (out_subblock_h * out_subblock_w) / 8.0)

    def _score_backend_preference(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score based on backend preference."""
        if candidate.backend == "matmul":
            return 0.8
        return 0.5

    def _score_fidelity_cost(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score based on math fidelity efficiency — lower fidelity = higher score."""
        fidelity = getattr(candidate, "math_fidelity", None)
        if fidelity is None or not isinstance(fidelity, MathFidelity):
            fidelity = features.get("math_fidelity_default")
        if fidelity is None or not isinstance(fidelity, MathFidelity):
            return 0.5

        cycle_cost = CYCLES_PER_TILE.get(fidelity, MAX_CYCLES_PER_TILE)
        return 1.0 - (0.75 * (cycle_cost - 16) / (MAX_CYCLES_PER_TILE - 16))

    def _score_production_bonus(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score bonus for production-derived candidates."""
        return 1.0 if candidate.params.get("production_derived", False) else 0.0
