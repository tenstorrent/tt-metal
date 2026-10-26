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
    extra_program=None,
    extra_tensors=(),
    staging: str | None = None,
    balance_lines: bool = False,
    w_prefetch=None,
) -> ttnn.Tensor:
    """sum over the cluster_axis group of input_tensor @ weight, scattered along scatter_dim (-1 or -2).

    input_tensor: (..., M, K), bfloat16 or bfloat8_b; weight: (K, N), bfloat16, bfloat8_b or bfloat4_b; both TILE,
    interleaved in DRAM or L1, M, K, N multiples of 32 and the scattered extent a multiple of 32 x the group size.
    num_links: None uses every link a hop of the group has. The output is bfloat16 (memory_config: interleaved, default
    DRAM).
    extra_program(coord, free_cores) -> ProgramDescriptor | None merges another program onto the cores this op leaves
    idle on each device (its core ranges must not overlap this op's); extra_tensors are its I/O tensors.
    staging: "exact" (default: the per-mesh staging tensor is kept and resized to each call) or "ephemeral" (allocated
    per call and freed once enqueued).
    balance_lines (scatter_dim -1): the planner may give the n-lines of a block that does not split evenly unequal,
    per-block rotating widths, using every line of the matmul rectangle (the busiest core computes fewer tiles over the
    call; more matmul cores, so fewer idle cores for extra_program). Same per-tile K accumulation; the cross-device add
    order of some blocks can differ from the uniform split's, so the output matches it within BF16 rounding.
    w_prefetch: the WeightPrefetch (matmul_reduce_scatter_weight_prefetch) whose jobs the program just before this call
    ran: W's first windows are then already in L1 and entry 0 starts on them (ignored when this call's plan differs).
    Raises ValueError when it was made for another weight tensor or already passed to a call."""
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
    (kernel_lib/l1_prefetch_dataflow.hpp).
    Returns a WeightPrefetch to pass to that call as w_prefetch, or None when its plan cannot take one (windows other
    than 1 included: each window's job multicasts into a whole n-line, and two at once into one line can hang).

    Contract. The jobs write [l1_begin, l1_end), where the call's W circular buffer will be; nothing is allocated for
    it. The host op that runs them checks that the region is free while it runs (above its own circular buffers, below
    every L1 buffer). From then until the matmul_reduce_scatter call:
      - no other program may run on the mesh (its circular buffers start at the same allocator base), and
      - no L1 buffer may be allocated in the region and written (the landed windows would be overwritten).
    Allocations in DRAM, deallocations and the call's own staging are fine. The call checks what it can: the prefetch
    is for its weight tensor, used once, and made for the plan the call gets (else it reads the windows itself); its
    reader credits them only if its W circular buffer is at l1_begin."""
    return _descriptor.weight_prefetch(
        weight,
        a_shape=list(input_shape),
        a_dtype=input_dtype,
        cluster_axis=cluster_axis,
        scatter_dim=_canonical_scatter_dim(scatter_dim, len(list(input_shape))),
        topology=topology,
        num_links=num_links,
        balance_lines=balance_lines,
        windows=windows,
    )
