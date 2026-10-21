# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""matmul_reduce_scatter — fused multi-device matmul + reduce-scatter (SUM) over one mesh axis, as one native C++ op.

Every device of a group holds a K-slice A[g] (..., M, K) and W[g] (K, N). The op computes, block by block in transport
order, each device's partial A[g] @ W[g] on a 2D dual-multicast matmul grid; the partial of block b travels along the
shorter way to its owner (ring) or along the line, landing straight in the L1 of the matmul core that owns the same
output tiles on the next chip, which adds its own partial and hands the sum to a port core that forwards it. See
op_design.md for the full design; the planner lives in
ttnn/cpp/ttnn/operations/matmul_reduce_scatter/device/matmul_reduce_scatter_planner.cpp.
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """inputs[0] = A shape (..., M, K), inputs[1] = W shape (K, N)."""
    a_shape, w_shape = inputs[0], inputs[1]
    M, K, N = a_shape[-2], a_shape[-1], w_shape[-1]
    if M % 32 == 0 and K % 32 == 0 and N % 32 == 0:
        return "tile_aligned"
    return "non_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "weight_dtype": [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "cluster_axis": [0, 1],
    "scatter_dim": [-1, -2],
    "topology": [Topology.Linear, Topology.Ring],
    "num_links": [1, 2],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


def default_compute_kernel_config():
    """Phase 0 default: HiFi2, fp32 DEST accumulation."""
    return ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)


def _canonical_scatter_dim(scatter_dim, rank):
    return scatter_dim - rank if scatter_dim >= 0 else scatter_dim


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(
    input_tensor,
    weight,
    *,
    cluster_axis,
    scatter_dim=-2,
    topology=Topology.Linear,
    num_links=None,
):
    a_shape = tuple(input_tensor.shape)
    w_shape = tuple(weight.shape)
    axes = {
        "dtype": input_tensor.dtype,
        "weight_dtype": weight.dtype,
        "layout": input_tensor.layout if input_tensor.layout != ttnn.TILE_LAYOUT else weight.layout,
        "cluster_axis": cluster_axis,
        "scatter_dim": _canonical_scatter_dim(scatter_dim, len(a_shape)),
        "topology": topology,
    }
    if num_links is not None:  # None is not a registry value: it resolves to all usable links after validate
        axes["num_links"] = num_links
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((a_shape, w_shape), axes)

    for axis, allowed in SUPPORTED.items():
        if axis in axes and axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"matmul_reduce_scatter: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"matmul_reduce_scatter: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# The registered C++ operation (ttnn/cpp/ttnn/operations/matmul_reduce_scatter): it checks this support contract,
# refuses with the same exceptions, plans and dispatches the program natively.
matmul_reduce_scatter = ttnn.matmul_reduce_scatter
