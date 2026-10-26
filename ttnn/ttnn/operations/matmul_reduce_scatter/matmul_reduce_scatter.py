# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""matmul_reduce_scatter — fused multi-device matmul + reduce-scatter (SUM) over one mesh axis, as ONE generic_op.

Every device of a group holds a K-slice A[g] (..., M, K) and W[g] (K, N). The op computes, block by block in transport
order, each device's partial A[g] @ W[g] on a 2D dual-multicast matmul grid; the partial of block b travels along the
shorter way to its owner (ring) or along the line, landing straight in the L1 of the matmul core that owns the same
output tiles on the next chip, which adds its own partial and hands the sum to a port core that forwards it. See
op_design.md for the full design; the planner lives in matmul_reduce_scatter_program_descriptor.py.
"""

from __future__ import annotations

import sys

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


def matmul_reduce_scatter(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    cluster_axis: int,
    scatter_dim: int = -2,
    topology=Topology.Linear,
    num_links: int | None = None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: ttnn.MemoryConfig | None = None,
    extra_program=None,
    extra_tensors=(),
    staging: str | None = None,
    balance_lines: bool = False,
    w_prefetch=None,
) -> ttnn.Tensor:
    """extra_program(coord, free_cores) -> ProgramDescriptor | None merges another program onto the cores this op
    leaves idle on each device (its core ranges must not overlap this op's); extra_tensors are its I/O tensors.
    staging: "exact", "grow" or "ephemeral" (allocated per call and freed once enqueued); default MRS_STAGING.
    balance_lines (scatter_dim -1): the planner may give the n-lines of a block that does not split evenly unequal,
    per-block rotating widths, using every line of the matmul rectangle (the busiest core computes fewer tiles over the
    call; more matmul cores, so fewer idle cores for extra_program). Same per-tile K accumulation; the cross-device add
    order of some blocks can differ from the uniform split's, so the output matches it within BF16 rounding.
    w_prefetch: the WeightPrefetch (matmul_reduce_scatter_weight_prefetch) whose jobs the previous program ran: W's
    first windows are then already in L1 and entry 0 starts on them (ignored when this call's plan differs)."""
    validate(
        input_tensor,
        weight,
        cluster_axis=cluster_axis,
        scatter_dim=scatter_dim,
        topology=topology,
        num_links=num_links,
    )
    from .matmul_reduce_scatter_program_descriptor import run

    pkg = sys.modules[__package__]  # test hooks are read from / written to the package at call time
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    return run(
        input_tensor,
        weight,
        cluster_axis=cluster_axis,
        scatter_dim=_canonical_scatter_dim(scatter_dim, len(tuple(input_tensor.shape))),
        topology=topology,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        hooks=pkg,
        extra_program=extra_program,
        extra_tensors=tuple(extra_tensors),
        staging=staging,
        balance_lines=balance_lines,
        w_prefetch=w_prefetch,
    )


def matmul_reduce_scatter_weight_prefetch(
    input_shape,
    input_dtype,
    weight: ttnn.Tensor,
    *,
    cluster_axis: int,
    scatter_dim: int = -2,
    topology=Topology.Linear,
    num_links: int | None = None,
    balance_lines: bool = False,
    windows: int = 1,
):
    """The jobs that land W's first `windows` K-block windows of a later matmul_reduce_scatter(A, weight, ...) call
    (A of input_shape / input_dtype, the same keywords) in L1, for the program just before it to run
    (kernel_lib/l1_prefetch_dataflow.hpp; e.g. ttnn.experimental.kda.sigmoid_gated_rms_norm's prefetch_* arguments).
    Returns a WeightPrefetch to pass to that call as w_prefetch, or None when its plan cannot take one."""
    from .matmul_reduce_scatter_program_descriptor import weight_prefetch

    return weight_prefetch(
        weight,
        a_shape=list(input_shape),
        a_dtype=input_dtype,
        cluster_axis=cluster_axis,
        scatter_dim=_canonical_scatter_dim(scatter_dim, len(list(input_shape))),
        topology=topology,
        num_links=num_links,
        hooks=sys.modules[__package__],
        balance_lines=balance_lines,
        windows=windows,
    )
