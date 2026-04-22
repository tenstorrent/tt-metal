# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul configuration search space definition and constraint validation.

This module defines the space of valid matmul configurations for Tenstorrent
hardware. It encodes hardware constraints (L1 memory, tile sizes, subblock
limits) and provides utilities to enumerate or validate candidate configs.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

TILE_SIZE = 32  # Default tile dimension

# Subblock choices ordered by preference (largest product first)
SUBBLOCK_HW_CHOICES = [
    (4, 2), (2, 4), (8, 1), (1, 8),
    (7, 1), (1, 7),
    (3, 2), (2, 3), (6, 1), (1, 6),
    (5, 1), (1, 5),
    (2, 2), (4, 1), (1, 4),
    (3, 1), (1, 3),
    (2, 1), (1, 2),
    (1, 1),
]


@dataclass
class MatmulShape:
    """Describes the logical shape of a matmul operation."""
    M: int
    K: int
    N: int
    batch_size: int = 1

    @property
    def M_tiles(self) -> int:
        return math.ceil(self.M / TILE_SIZE)

    @property
    def K_tiles(self) -> int:
        return math.ceil(self.K / TILE_SIZE)

    @property
    def N_tiles(self) -> int:
        return math.ceil(self.N / TILE_SIZE)


@dataclass
class DeviceConstraints:
    """Hardware constraints for a Tenstorrent device."""
    grid_x: int = 8
    grid_y: int = 8
    max_l1_bytes: int = 1_499_136  # ~1.4 MB L1 per core (Wormhole)
    tile_size_bytes: int = 2048  # bfloat16 tile = 32*32*2
    fp32_dest_acc_en: bool = False

    @property
    def num_cores(self) -> int:
        return self.grid_x * self.grid_y

    @property
    def subblock_max_product(self) -> int:
        return 4 if self.fp32_dest_acc_en else 8


@dataclass
class MatmulConfig:
    """A candidate matmul configuration."""
    M_block_size: int
    K_block_size: int
    N_block_size: int
    subblock_h: int = 2
    subblock_w: int = 2
    grid_x: int = 8
    grid_y: int = 8
    use_1d_systolic: bool = False
    mcast_in0: bool = True
    fuse_batch: bool = False

    def to_tuple(self) -> Tuple[int, int, int, int, int]:
        return (self.M_block_size, self.K_block_size, self.N_block_size,
                self.subblock_h, self.subblock_w)


def get_valid_subblock(per_core_M: int, per_core_N: int,
                       fp32_dest_acc_en: bool = False) -> Tuple[int, int]:
    """Select the best subblock dimensions for given per-core tile counts."""
    max_product = 4 if fp32_dest_acc_en else 8
    for sh, sw in SUBBLOCK_HW_CHOICES:
        if sh * sw > max_product:
            continue
        if per_core_M % sh == 0 and per_core_N % sw == 0:
            return (sh, sw)
    return (1, 1)


def estimate_l1_usage(M_block: int, K_block: int, N_block: int,
                      tile_bytes: int = 2048,
                      buffering_depth: int = 2) -> int:
    """Estimate L1 memory usage for a matmul configuration.

    Accounts for input A CB, input B CB, output CB, and intermediate CB.
    """
    in0_cb = M_block * K_block * tile_bytes * buffering_depth
    in1_cb = K_block * N_block * tile_bytes * buffering_depth
    out_cb = M_block * N_block * tile_bytes
    interm_cb = M_block * N_block * tile_bytes  # intermediate accumulation
    return in0_cb + in1_cb + out_cb + interm_cb


def is_config_valid(config: MatmulConfig, shape: MatmulShape,
                    constraints: DeviceConstraints) -> bool:
    """Check if a matmul config is valid given shape and device constraints."""
    M_tiles = shape.M_tiles
    K_tiles = shape.K_tiles
    N_tiles = shape.N_tiles

    # Block sizes must be positive
    if config.M_block_size < 1 or config.K_block_size < 1 or config.N_block_size < 1:
        return False

    # Subblock must divide block dimensions
    if config.M_block_size % config.subblock_h != 0:
        return False
    if config.N_block_size % config.subblock_w != 0:
        return False

    # Subblock product constraint
    if config.subblock_h * config.subblock_w > constraints.subblock_max_product:
        return False

    # K_block must divide K_tiles
    if K_tiles % config.K_block_size != 0:
        # Allow if K_block_size <= K_tiles (will pad)
        if config.K_block_size > K_tiles:
            return False

    # L1 memory constraint
    l1_usage = estimate_l1_usage(
        config.M_block_size, config.K_block_size, config.N_block_size,
        constraints.tile_size_bytes)
    if l1_usage >= constraints.max_l1_bytes:
        return False

    return True


def enumerate_candidate_configs(shape: MatmulShape,
                                constraints: DeviceConstraints,
                                max_block: int = 64) -> List[MatmulConfig]:
    """Generate all valid candidate configurations for a given shape and device.

    Uses smart pruning to keep the search space manageable.
    """
    M_tiles = shape.M_tiles
    K_tiles = shape.K_tiles
    N_tiles = shape.N_tiles

    # Candidate block sizes: divisors + powers of 2 + common values
    def get_candidates(num_tiles: int) -> List[int]:
        base = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20}
        # Add all divisors up to max_block
        for i in range(1, min(num_tiles + 1, max_block + 1)):
            if num_tiles % i == 0:
                base.add(i)
        # Add powers of 2
        p = 1
        while p <= min(num_tiles, max_block):
            base.add(p)
            p *= 2
        return sorted(b for b in base if b <= min(num_tiles, max_block))

    m_candidates = get_candidates(M_tiles)
    k_candidates = get_candidates(K_tiles)
    n_candidates = get_candidates(N_tiles)

    valid_configs = []
    for m_block in m_candidates:
        for k_block in k_candidates:
            for n_block in n_candidates:
                sub_h, sub_w = get_valid_subblock(
                    m_block, n_block, constraints.fp32_dest_acc_en)
                cfg = MatmulConfig(
                    M_block_size=m_block,
                    K_block_size=k_block,
                    N_block_size=n_block,
                    subblock_h=sub_h,
                    subblock_w=sub_w,
                    grid_x=constraints.grid_x,
                    grid_y=constraints.grid_y,
                )
                if is_config_valid(cfg, shape, constraints):
                    valid_configs.append(cfg)

    return valid_configs
