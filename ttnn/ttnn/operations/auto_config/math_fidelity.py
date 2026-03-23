# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Math fidelity modeling for matmul auto-config.

Math Fidelity defines the number of phases used to compute a high-precision
multiplication using four low-precision multiplications. Higher fidelity
means more accurate but more compute cycles per tile.

Cycle cost per tile (from matrix engine tech report):
    LoFi:  16 cycles — SrcA MSBs × SrcB MSBs
    HiFi2: 32 cycles — + SrcA LSBs × SrcB MSBs
    HiFi3: 48 cycles — + SrcA MSBs × SrcB LSBs
    HiFi4: 64 cycles — + SrcA LSBs × SrcB LSBs

Dtype-to-fidelity mapping (from tech report + PR #39628):
    - Bfp8:  LoFi + HiFi2 fully consumes all mantissa bits.
    - Bf16:  Needs all 4 phases, but HiFi3/HiFi4 can sometimes be skipped.
    - Bf16 × Bfp4: HiFi3 preferred over HiFi2 (PR #39628), because bfp4
      SrcB has very few mantissa bits — HiFi2 re-reads SrcB MSBs (no new
      info), while HiFi3 reads SrcB LSBs (new info).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class MathFidelity(IntEnum):
    """Math fidelity levels for Tenstorrent matrix engines."""
    LoFi  = 0   # 16 cycles/tile — SrcA MSBs × SrcB MSBs
    HiFi2 = 1   # 32 cycles/tile — + SrcA LSBs × SrcB MSBs
    HiFi3 = 2   # 48 cycles/tile — + SrcA MSBs × SrcB LSBs
    HiFi4 = 3   # 64 cycles/tile — + SrcA LSBs × SrcB LSBs


# Cycle cost per tile for each fidelity level
CYCLES_PER_TILE: Dict[MathFidelity, int] = {
    MathFidelity.LoFi:  16,
    MathFidelity.HiFi2: 32,
    MathFidelity.HiFi3: 48,
    MathFidelity.HiFi4: 64,
}

# Normalization constant: max cycles per tile is HiFi4 = 64
MAX_CYCLES_PER_TILE = 64


# ===========================================================================
# Dtype → fidelity constraint table
# ===========================================================================
# Maps (srcA_dtype_key, srcB_dtype_key) to the list of valid fidelities,
# ordered from most efficient to most accurate.
#
# Dtype keys are the suffixes from ttnn DataType strings, e.g.:
#   "DataType.BFLOAT16" → "BFLOAT16"
#   "DataType.BFLOAT8_B" → "BFLOAT8_B"
#   "DataType.BFLOAT4_B" → "BFLOAT4_B"

DTYPE_FIDELITY_CONSTRAINTS: Dict[Tuple[str, str], List[MathFidelity]] = {
    # Bfp8 × Bfp8: LoFi + HiFi2 consumes all mantissa bits
    ("BFLOAT8_B", "BFLOAT8_B"):   [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4],

    # Bfp8 × Bfp4: HiFi2 sufficient for bfp8 precision
    ("BFLOAT8_B", "BFLOAT4_B"):   [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4],

    # Bf16 × Bf16: needs all 4 phases for full precision
    ("BFLOAT16", "BFLOAT16"):     [MathFidelity.HiFi4],

    # Bf16 × Bfp8: HiFi2 catches the important SrcA LSBs × SrcB MSBs
    ("BFLOAT16", "BFLOAT8_B"):    [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4],

    # Bf16 × Bfp4 — KEY CASE from PR #39628:
    #   Bfp4 SrcB has ~3 mantissa bits, all in MSBs.
    #   LoFi:  SrcA_MSB × SrcB_MSB ← useful
    #   HiFi2: SrcA_LSB × SrcB_MSB ← marginal (SrcB MSBs already consumed)
    #   HiFi3: SrcA_MSB × SrcB_LSB ← useful (new SrcB bits)
    #   Therefore, prefer HiFi3 over HiFi2 for bf16 × bfp4.
    ("BFLOAT16", "BFLOAT4_B"):    [MathFidelity.HiFi3, MathFidelity.HiFi4],

    # Bfp4 × Bfp4: LoFi is sufficient
    ("BFLOAT4_B", "BFLOAT4_B"):   [MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4],

    # Bfp4 × Bfp8: LoFi + HiFi2 covers it
    ("BFLOAT4_B", "BFLOAT8_B"):   [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4],

    # Bfp4 × Bf16: mirror of the above
    ("BFLOAT4_B", "BFLOAT16"):    [MathFidelity.HiFi3, MathFidelity.HiFi4],

    # Bfp8 × Bf16: mirror
    ("BFLOAT8_B", "BFLOAT16"):    [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4],
}

# Default fidelity ordering when dtype pair is unknown
_DEFAULT_FIDELITIES = [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4]


def _normalize_dtype_key(dtype_str: str) -> str:
    """Extract the dtype suffix from a ttnn DataType string.

    "DataType.BFLOAT16" → "BFLOAT16"
    "bfloat16" → "BFLOAT16"
    """
    if "." in dtype_str:
        dtype_str = dtype_str.split(".")[-1]
    return dtype_str.upper()


def valid_fidelities(dtype_a: str, dtype_b: str) -> List[MathFidelity]:
    """Return the list of valid math fidelities for a dtype pair.

    Args:
        dtype_a: String representation of input A dtype (e.g. "DataType.BFLOAT16").
        dtype_b: String representation of input B dtype (e.g. "DataType.BFLOAT8_B").

    Returns:
        Ordered list of valid MathFidelity levels (best efficiency first).
    """
    key_a = _normalize_dtype_key(dtype_a)
    key_b = _normalize_dtype_key(dtype_b)
    return DTYPE_FIDELITY_CONSTRAINTS.get((key_a, key_b), _DEFAULT_FIDELITIES)


def default_fidelity(dtype_a: str, dtype_b: str) -> MathFidelity:
    """Return the recommended default fidelity for a dtype pair.

    Picks the minimum fidelity that fully consumes the available mantissa bits.
    """
    fidelities = valid_fidelities(dtype_a, dtype_b)
    return fidelities[0] if fidelities else MathFidelity.HiFi4


def fidelity_cycle_cost(fidelity: MathFidelity) -> int:
    """Return the cycle cost per tile for a given fidelity level."""
    return CYCLES_PER_TILE[fidelity]


def fidelity_to_ttnn_string(fidelity: MathFidelity) -> str:
    """Convert MathFidelity enum to the ttnn MathFidelity string.

    This is used when constructing compute_config for ttnn.matmul.
    """
    return {
        MathFidelity.LoFi:  "MathFidelity.LoFi",
        MathFidelity.HiFi2: "MathFidelity.HiFi2",
        MathFidelity.HiFi3: "MathFidelity.HiFi3",
        MathFidelity.HiFi4: "MathFidelity.HiFi4",
    }[fidelity]


# ===========================================================================
# GPT-style training shapes (for training data expansion)
# ===========================================================================

GPT_ATTENTION_SHAPES = [
    # (M, K, N, description)
    # QK^T: [seq, heads*head_dim] × [heads*head_dim, seq]
    (2048, 4096, 2048, "GPT-2 large QK^T, seq=2048"),
    (128,  4096, 128,  "GPT decode QK^T, short seq"),
    (1,    4096, 2048, "Single-token decode QK^T"),
    # V projection: [seq, seq] × [seq, head_dim]
    (2048, 2048, 128,  "GPT V projection"),
    # MLP: [seq, hidden] × [hidden, 4*hidden]
    (2048, 4096, 16384, "GPT MLP up-proj, seq=2048"),
    (128,  4096, 16384, "GPT MLP up-proj, seq=128"),
    # Multi-head with >32 heads (PR #39196 pattern)
    (2048, 8192, 2048,  "64 heads × 128 dim attention"),
    # Small batch decode (PR #39120 tile-rounding)
    (1,    4096, 4096,  "Decode 1-token 4K hidden"),
    (8,    4096, 4096,  "Decode 8-token 4K hidden"),
    (16,   4096, 11008, "LLaMA MLP decode"),
    # Non-tile-aligned vocabulary (PR #39296 padded vocab)
    (32,   4096, 32000, "LLaMA vocab head batch=32"),
    (32,   4096, 128256, "Large vocab head batch=32"),
    # Additional GPT-OSS pipeline parallelism shapes (sankarmanoj-tt)
    (256,  4096, 4096,  "Pipeline parallel mid-seq"),
    (512,  4096, 4096,  "Pipeline parallel mid-seq 512"),
    (1024, 4096, 4096,  "Pipeline parallel mid-seq 1024"),
    # DeepSeek V3 patterns (yieldthought)
    (2048, 7168, 2048,  "DeepSeek V3 attention"),
    (2048, 7168, 18432, "DeepSeek V3 MLP"),
]
