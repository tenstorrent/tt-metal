# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Math fidelity modeling for matmul auto-config."""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Tuple


class MathFidelity(IntEnum):
    """Math fidelity levels for Tenstorrent matrix engines."""

    LoFi = 0
    HiFi2 = 1
    HiFi3 = 2
    HiFi4 = 3


CYCLES_PER_TILE: Dict[MathFidelity, int] = {
    MathFidelity.LoFi: 16,
    MathFidelity.HiFi2: 32,
    MathFidelity.HiFi3: 48,
    MathFidelity.HiFi4: 64,
}

MAX_CYCLES_PER_TILE = 64

DTYPE_FIDELITY_CONSTRAINTS: Dict[Tuple[str, str], List[MathFidelity]] = {
    ("BFLOAT8_B", "BFLOAT8_B"): [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4],
    ("BFLOAT8_B", "BFLOAT4_B"): [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4],
    ("BFLOAT16", "BFLOAT16"): [MathFidelity.HiFi4],
    ("BFLOAT16", "BFLOAT8_B"): [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4],
    ("BFLOAT16", "BFLOAT4_B"): [MathFidelity.HiFi3, MathFidelity.HiFi4],
    ("BFLOAT4_B", "BFLOAT4_B"): [MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4],
    ("BFLOAT4_B", "BFLOAT8_B"): [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4],
    ("BFLOAT4_B", "BFLOAT16"): [MathFidelity.HiFi3, MathFidelity.HiFi4],
    ("BFLOAT8_B", "BFLOAT16"): [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4],
}

_DEFAULT_FIDELITIES = [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4]


def _normalize_dtype_key(dtype_str: str) -> str:
    """Extract the dtype suffix from a ttnn DataType string."""
    if "." in dtype_str:
        dtype_str = dtype_str.split(".")[-1]
    return dtype_str.upper()


def valid_fidelities(dtype_a: str, dtype_b: str) -> List[MathFidelity]:
    """Return valid math fidelities for a dtype pair (best efficiency first)."""
    return DTYPE_FIDELITY_CONSTRAINTS.get(
        (_normalize_dtype_key(dtype_a), _normalize_dtype_key(dtype_b)), _DEFAULT_FIDELITIES
    )


def default_fidelity(dtype_a: str, dtype_b: str) -> MathFidelity:
    """Return the recommended default fidelity for a dtype pair."""
    fidelities = valid_fidelities(dtype_a, dtype_b)
    return fidelities[0] if fidelities else MathFidelity.HiFi4


def fidelity_cycle_cost(fidelity: MathFidelity) -> int:
    """Return the cycle cost per tile for a given fidelity level."""
    return CYCLES_PER_TILE[fidelity]


def fidelity_to_ttnn_string(fidelity: MathFidelity) -> str:
    """Convert MathFidelity enum to the ttnn MathFidelity string."""
    return f"MathFidelity.{fidelity.name}"
