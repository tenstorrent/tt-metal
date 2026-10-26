# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""matmul_reduce_scatter: fused multi-device matmul + reduce-scatter (SUM) over one mesh axis, as ONE generic_op.

Every device of a group holds a K-slice A[g] (..., M, K) and W[g] (K, N). The op computes, block by block in transport
order, each device's partial A[g] @ W[g] on a 2D dual-multicast matmul grid; the partial of block b travels along the
shorter way to its owner (ring) or along the line, landing straight in the L1 of the matmul core that owns the same
output tiles on the next chip, which adds its own partial and hands the sum to a port core that forwards it. See
README.md for the design; the planner lives in matmul_reduce_scatter_program_descriptor.py.
"""

from __future__ import annotations

import ttnn
from ttnn.operations.ccl import Topology

from . import matmul_reduce_scatter_program_descriptor as _descriptor


def default_compute_kernel_config():
    """HiFi2, fp32 DEST accumulation."""
    return ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)


def _canonical_scatter_dim(scatter_dim, rank):
    return scatter_dim - rank if scatter_dim >= 0 else scatter_dim


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
    staging: str | None = None,
) -> ttnn.Tensor:
    """sum over the cluster_axis group of input_tensor @ weight, scattered along scatter_dim (-1 or -2).

    input_tensor: (..., M, K), bfloat16 or bfloat8_b; weight: (K, N), bfloat16, bfloat8_b or bfloat4_b; both TILE,
    interleaved in DRAM or L1, M, K, N multiples of 32 and the scattered extent a multiple of 32 x the group size.
    num_links: None uses every link a hop of the group has. The output is bfloat16 (memory_config: interleaved, default
    DRAM).
    staging: "exact" (default: the per-mesh staging tensor is kept and resized to each call) or "ephemeral" (allocated
    per call and freed once enqueued)."""
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    return _descriptor.run(
        input_tensor,
        weight,
        cluster_axis=cluster_axis,
        scatter_dim=_canonical_scatter_dim(scatter_dim, len(tuple(input_tensor.shape))),
        topology=topology,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        staging=staging,
    )
