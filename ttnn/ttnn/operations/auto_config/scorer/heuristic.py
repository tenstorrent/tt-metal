# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Heuristic performance scorer for matmul config candidates.

Scores candidates based on:
- Compute utilization (how many cores are effectively used)
- Memory bandwidth efficiency (L1 vs DRAM access patterns)
- Data movement overhead (multicast direction alignment)
- Config-input layout alignment
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ttnn.operations.auto_config.base import ConfigCandidate

logger = logging.getLogger(__name__)

TILE_SIZE = 32


class HeuristicScorer:
    """
    Rule-based performance scorer for matmul configurations.

    Higher scores indicate better expected performance. The scorer considers:
    1. Core utilization: more effectively used cores = higher score
    2. Block size efficiency: larger blocks = fewer iterations = higher throughput
    3. Memory layout alignment: matching input sharding to config pattern
    4. Subblock efficiency: larger subblocks = better register utilization
    """

    def __init__(self):
        # Weight factors for each scoring component
        self.w_utilization = 0.35
        self.w_block_efficiency = 0.25
        self.w_layout_alignment = 0.20
        self.w_subblock_efficiency = 0.10
        self.w_backend_preference = 0.10

    def score(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score a candidate configuration. Returns value in [0, 1]."""
        scores = {
            "utilization": self._score_utilization(candidate, features),
            "block_efficiency": self._score_block_efficiency(candidate, features),
            "layout_alignment": self._score_layout_alignment(candidate, features),
            "subblock_efficiency": self._score_subblock_efficiency(candidate, features),
            "backend_preference": self._score_backend_preference(candidate, features),
        }

        total = (
            self.w_utilization * scores["utilization"]
            + self.w_block_efficiency * scores["block_efficiency"]
            + self.w_layout_alignment * scores["layout_alignment"]
            + self.w_subblock_efficiency * scores["subblock_efficiency"]
            + self.w_backend_preference * scores["backend_preference"]
        )

        logger.debug(
            f"Scored {candidate.config_family}: {total:.3f} "
            f"(util={scores['utilization']:.2f}, block={scores['block_efficiency']:.2f}, "
            f"layout={scores['layout_alignment']:.2f}, subblk={scores['subblock_efficiency']:.2f}, "
            f"backend={scores['backend_preference']:.2f})"
        )

        return total

    def _score_utilization(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score based on how many cores are effectively used."""
        num_cores = features["num_cores"]
        M_tiles = features["M_tiles"]
        N_tiles = features["N_tiles"]
        batch_size = features["batch_size_a"]
        total_output_tiles = batch_size * M_tiles * N_tiles

        config = candidate.config
        if candidate.config_family in ("DRAMSharded", "BatchedDRAMSharded", "MultiCore"):
            # DRAM-sharded uses all DRAM banks, so utilization is typically good
            return 0.7

        per_core_M = getattr(config, "per_core_M", M_tiles)
        per_core_N = getattr(config, "per_core_N", N_tiles)

        if per_core_M <= 0 or per_core_N <= 0:
            return 0.1

        tiles_per_core = per_core_M * per_core_N
        if tiles_per_core <= 0:
            return 0.1

        # Approximate number of cores needed
        estimated_cores = max(1, total_output_tiles / tiles_per_core)
        # Utilization = fraction of device cores used
        utilization = min(1.0, estimated_cores / num_cores)

        return utilization

    def _score_block_efficiency(self, candidate: ConfigCandidate, features: Dict[str, Any]) -> float:
        """Score based on block sizes — larger blocks mean fewer iterations."""
        K_tiles = features["K_tiles"]

        in0_block_w = candidate.params.get("in0_block_w", 1)
        per_core_M = candidate.params.get("per_core_M", 1)
        per_core_N = candidate.params.get("per_core_N", 1)

        # Larger in0_block_w relative to K means fewer inner loop iterations
        k_efficiency = min(1.0, in0_block_w / max(1, K_tiles))

        # Larger output blocks per core means less overhead
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

        # Perfect alignment bonuses
        if config_family == "MultiCast1D":
            if mcast_in0 and is_wide:
                return 1.0  # Wide shape + mcast_in0 = ideal
            if not mcast_in0 and is_tall:
                return 1.0  # Tall shape + mcast_in1 = ideal
            # Shape doesn't match mcast direction
            return 0.5

        if config_family == "MultiCast2D":
            if not is_tall and not is_wide:
                return 1.0  # Square-ish shapes are great for 2D mcast
            return 0.7

        if config_family == "DRAMSharded":
            if not is_a_sharded:
                return 0.8  # Good when inputs are already in DRAM
            return 0.3

        if config_family == "Reuse":
            if features["is_batched_b"]:
                return 1.0  # Reuse is the right choice for batched B
            return 0.3

        if config_family == "MinimalMatmul":
            return 0.6  # Generally competitive, slightly less configurable

        return 0.5

    def _score_subblock_efficiency(
        self, candidate: ConfigCandidate, features: Dict[str, Any]
    ) -> float:
        """Score based on subblock sizes — larger subblocks use registers better."""
        out_subblock_h = candidate.params.get("out_subblock_h", 1)
        out_subblock_w = candidate.params.get("out_subblock_w", 1)

        subblock_product = out_subblock_h * out_subblock_w
        # Max useful subblock product is 8 (DST register limit)
        return min(1.0, subblock_product / 8.0)

    def _score_backend_preference(
        self, candidate: ConfigCandidate, features: Dict[str, Any]
    ) -> float:
        """Score based on backend preference."""
        if candidate.backend == "matmul":
            return 0.8  # Standard backend, well-tested
        if candidate.backend == "minimal_matmul":
            return 0.6  # Experimental, may have better throughput for specific shapes
        return 0.5
