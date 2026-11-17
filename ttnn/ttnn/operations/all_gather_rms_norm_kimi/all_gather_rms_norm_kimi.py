# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_gather_rms_norm_kimi — a fused Fabric ring all-gather + RMSNorm in ONE ttnn.generic_op.

Every device of a cluster_axis=1 ring group of G devices holds its (1, 1, M, W_local) slice of the hidden dim and
returns the RMS-normalized full (1, 1, M, W) row block, W = G * W_local. One MeshProgramDescriptor, one program per
chip (see op_design.md and all_gather_rms_norm_kimi_program_descriptor.py).
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology

from .all_gather_rms_norm_kimi_program_descriptor import (
    build_geometry,
    create_mesh_program_descriptor,
    get_resources,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Per-device INPUT shape: (..., H, W_local)."""
    shape = inputs[0]
    h, w = shape[-2], shape[-1]
    if w % 32 != 0:
        return "w_non_aligned"
    if h % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(inputs[0])


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "rank": [4],
    "gamma_dtype": [ttnn.bfloat16],
    "gamma_layout": [ttnn.ROW_MAJOR_LAYOUT],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
    "cluster_axis": [1],
    "topology": [Topology.Ring],
    "num_links": [2],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# Default compute config (single source of truth; the golden axis tagger reads it too)
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(
    input_tensor,
    *,
    gamma,
    cluster_axis,
    epsilon=1e-6,
    topology=Topology.Ring,
    num_links=2,
    compute_kernel_config=None,
):
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "layout": input_tensor.layout,
        "gamma_dtype": gamma.dtype,
        "gamma_layout": gamma.layout,
        "memory_layout": input_tensor.memory_config().memory_layout,
        "cluster_axis": cluster_axis,
        "topology": topology,
        "num_links": num_links,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"all_gather_rms_norm_kimi: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"all_gather_rms_norm_kimi: unsupported combination (refinement candidate): {exc}")


def _check_caller_errors(input_tensor, gamma, cluster_axis, num_links, mesh_device):
    """Caller errors (ValueError), not support refusals."""
    mesh_shape = tuple(mesh_device.shape)
    G = mesh_shape[cluster_axis]
    if G == 1:
        raise ValueError(f"all_gather_rms_norm_kimi: mesh axis {cluster_axis} has size 1 (mesh {mesh_shape})")
    if G < 3:
        raise ValueError(f"all_gather_rms_norm_kimi: a Ring needs at least 3 devices along cluster_axis, got {G}")
    if num_links < 1:
        raise ValueError(f"all_gather_rms_norm_kimi: num_links must be >= 1, got {num_links}")
    shape = tuple(input_tensor.shape)
    if shape[-1] % 32 != 0 or shape[-2] % 32 != 0:
        raise ValueError(f"all_gather_rms_norm_kimi: input must be tile-aligned, got {shape}")
    W = G * shape[-1]
    want = [1, 1, -(-W // 32), 32]
    got = list(gamma.padded_shape)
    if got != want or list(gamma.shape) != want:
        raise ValueError(
            f"all_gather_rms_norm_kimi: gamma must be the ROW_MAJOR stick format (1, 1, ceil(W/32), 32) = {want} "
            f"for W = {W}, got gamma shape {list(gamma.shape)}"
        )
    for t, what in ((input_tensor, "input"), (gamma, "gamma")):
        mc = t.memory_config()
        if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise ValueError(f"all_gather_rms_norm_kimi: {what} must be DRAM interleaved")
    # Every hop of every group, including the wrap hop, needs num_links forwarding links under the active config.
    rows, cols = mesh_shape
    node = lambda r, c: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
    groups = (
        [[(r, c) for c in range(cols)] for r in range(rows)]
        if cluster_axis == 1
        else [[(r, c) for r in range(rows)] for c in range(cols)]
    )
    for grp in groups:
        for i, a in enumerate(grp):
            for b in (grp[(i + 1) % G], grp[(i - 1) % G]):
                try:
                    links = ttnn.get_forwarding_link_indices(node(*a), node(*b))
                except Exception as e:  # noqa: BLE001
                    raise ValueError(
                        f"all_gather_rms_norm_kimi: no forwarding link {a} -> {b} "
                        f"(fabric {ttnn.get_fabric_config()}): {e}"
                    ) from e
                if len(links) < num_links:
                    raise ValueError(
                        f"all_gather_rms_norm_kimi: hop {a} -> {b} has {len(links)} usable link(s), {num_links} "
                        f"requested (Ring needs every hop incl. the wrap; fabric {ttnn.get_fabric_config()})"
                    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def all_gather_rms_norm_kimi(
    input_tensor: ttnn.Tensor,
    *,
    gamma: ttnn.Tensor,
    cluster_axis: int,
    epsilon: float = 1e-6,
    topology=Topology.Ring,
    num_links: int = 2,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    validate(
        input_tensor,
        gamma=gamma,
        cluster_axis=cluster_axis,
        epsilon=epsilon,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
    )
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    mesh_device = input_tensor.device()
    _check_caller_errors(input_tensor, gamma, cluster_axis, num_links, mesh_device)

    geo = build_geometry(mesh_device, tuple(input_tensor.padded_shape), cluster_axis, num_links)
    res = get_resources(mesh_device, geo)

    shape = list(input_tensor.shape)
    out_shape = shape[:-1] + [geo.G * shape[-1]]
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), input_tensor.dtype, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    desc = create_mesh_program_descriptor(geo, res, input_tensor, gamma, output, float(epsilon), cfg)
    return ttnn.generic_op([input_tensor, gamma, output], desc)
