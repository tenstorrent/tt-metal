# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""high_bw_all_reduce — bandwidth-optimized Fabric-2D all-reduce (SUM).

Registry model (see eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS,
validate(). The entry point dispatches exactly one ttnn.generic_op with a
MeshProgramDescriptor (one ProgramDescriptor per mesh coordinate) implementing
the chain schemes of op_design.md: R1 `chain_line` (axis line), R2 `chain_snake_line`
(cluster_axis=None, Linear) and R3 `rotated_chain_ring` (Ring).
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Per-device shape -> tile_aligned / w_non_aligned / h_non_aligned."""
    shape = inputs[0]
    h, w = shape[-2], shape[-1]
    if w % 32 != 0:
        return "w_non_aligned"
    if h % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "cluster_axis": [0, 1, None],
    "topology": [Topology.Linear, Topology.Ring],
    "num_links": [1, 2],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # Per-axis rings close over a torus wrap link (FABRIC_2D_TORUS_X/Y). The R3 kernels are
    # route-agnostic (the None snake ring runs them), but no torus cluster was available to
    # verify the wrap-edge route. Refused until one can run them.
    {"topology": Topology.Ring, "cluster_axis": 0},
    {"topology": Topology.Ring, "cluster_axis": 1},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(
    input_tensor,
    *,
    cluster_axis,
    topology=Topology.Linear,
    num_links=None,
    memory_config=None,
):
    """Registry gate: SUPPORTED per axis, then EXCLUSIONS. num_links=None is the
    default (resolves to every usable link) and skips its SUPPORTED check."""
    shape = tuple(input_tensor.shape)
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "cluster_axis": cluster_axis,
        "topology": topology,
        "num_links": num_links,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axis == "num_links" and axes[axis] is None:
            continue
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"high_bw_all_reduce: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"high_bw_all_reduce: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Caller-error checks (ValueError) — not support refusals.
# ---------------------------------------------------------------------------

_FABRIC_2D_CONFIGS = None


def _fabric_2d_configs():
    global _FABRIC_2D_CONFIGS
    if _FABRIC_2D_CONFIGS is None:
        # Every send is one hop, so 1D fabrics work too (the kernels route by hop count there).
        names = [
            "FABRIC_2D",
            "FABRIC_2D_TORUS_X",
            "FABRIC_2D_TORUS_Y",
            "FABRIC_2D_TORUS_XY",
            "FABRIC_1D",
            "FABRIC_1D_RING",
            "FABRIC_1D_NEIGHBOR_EXCHANGE",
        ]
        _FABRIC_2D_CONFIGS = {getattr(ttnn.FabricConfig, n) for n in names if hasattr(ttnn.FabricConfig, n)}
    return _FABRIC_2D_CONFIGS


def _is_dram_interleaved(memory_config):
    return (
        memory_config.buffer_type == ttnn.BufferType.DRAM
        and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    )


def _check_caller(input_tensor, memory_config):
    if len(input_tensor.shape) < 2:
        raise ValueError("high_bw_all_reduce: input rank must be >= 2")
    if not _is_dram_interleaved(input_tensor.memory_config()):
        raise ValueError("high_bw_all_reduce: input must be DRAM interleaved")
    if ttnn.get_fabric_config() not in _fabric_2d_configs():
        raise ValueError(
            f"high_bw_all_reduce: requires an active FABRIC_2D* or FABRIC_1D* fabric config, got {ttnn.get_fabric_config()}"
        )
    if memory_config is not None and not _is_dram_interleaved(memory_config):
        raise UnsupportedAxisValue(
            f"high_bw_all_reduce: memory_config={memory_config} not supported (DRAM interleaved only)"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def high_bw_all_reduce(
    input_tensor: ttnn.Tensor,
    *,
    cluster_axis,
    topology=Topology.Linear,
    num_links=None,
    memory_config=None,
) -> ttnn.Tensor:
    validate(
        input_tensor,
        cluster_axis=cluster_axis,
        topology=topology,
        num_links=num_links,
        memory_config=memory_config,
    )
    _check_caller(input_tensor, memory_config)

    from .high_bw_all_reduce_program_descriptor import build_and_dispatch

    return build_and_dispatch(input_tensor, cluster_axis=cluster_axis, topology=topology, num_links=num_links)
