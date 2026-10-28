# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""distributed_rms_norm — RMSNorm over a hidden dim sharded across one axis of a device mesh.

    out[p] = x[p] / sqrt(sum_g sum_w x[g]^2 / (G * W_local) + eps) * gamma[p]

One native dispatch per call: one ttnn.generic_op over one MeshProgramDescriptor (one program per chip). Each chip
runs the base op's local work (rms_norm, re-split for the grid left after the fabric port cores) in row waves; the
per-row fp32 Σx² partials are all-gathered over TT-Fabric by store-and-forward relay and summed in group-position
order on every chip (op_design.md).

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() drive both the runtime gate and the golden-test
xfail decisions.
"""

from __future__ import annotations

from typing import Optional

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology

from .distributed_rms_norm_program_descriptor import (
    collected_count,
    create_mesh_program_descriptor,
    forget_resources,
    discovered_links,
    plan_summary as _plan_summary,
)


# ---------------------------------------------------------------------------
# Default compute config — the ONE place the default is defined.
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------

TILE_DIM = 32


def tag_alignment(inputs, axes):
    shape = inputs[0]
    H, W = shape[-2], shape[-1]
    if W % TILE_DIM != 0:
        return "w_non_aligned"
    if H % TILE_DIM != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return int(len(inputs[0]))


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, "none"],
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    "memory_layout": [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
    "cluster_axis": [0, 1],
    "topology": [Topology.Linear, Topology.Ring],
    # None = "every usable link": validate() canonicalizes it (discovered_links) and gates on the canonical value, but
    # callers that omit num_links are tagged with the raw None (golden axes.py classify_call), so it is listed too.
    "num_links": [1, 2, None],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # Natively rejected precision combination (precision_convention.md).
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _per_device_shape(t):
    return list(t.shape)


def _hard_validate(input_tensor, cluster_axis, gamma, topology, num_links):
    shape = _per_device_shape(input_tensor)
    if len(shape) < 2:
        raise ValueError(f"distributed_rms_norm: input must have rank >= 2, got rank {len(shape)}")
    mesh = input_tensor.device()
    mesh_shape = tuple(mesh.shape)
    if cluster_axis not in (0, 1):
        raise ValueError(f"distributed_rms_norm: cluster_axis must be 0 or 1, got {cluster_axis}")
    if mesh_shape[cluster_axis] < 2:
        raise ValueError(f"distributed_rms_norm: mesh axis {cluster_axis} has size {mesh_shape[cluster_axis]}")
    W_local = shape[-1]
    if gamma is not None:
        g = list(gamma.shape)
        if gamma.layout == ttnn.TILE_LAYOUT:
            if g != [1, 1, 1, W_local]:
                raise ValueError(f"distributed_rms_norm: TILE gamma must be (1, 1, 1, {W_local}), got {g}")
        else:
            want = [1, 1, -(-W_local // TILE_DIM), TILE_DIM]
            if g != want:
                raise ValueError(f"distributed_rms_norm: ROW_MAJOR gamma must be sticks {tuple(want)}, got {g}")
    if ttnn.get_fabric_config() == ttnn.FabricConfig.DISABLED:
        raise RuntimeError("distributed_rms_norm: the fabric is disabled (open the mesh with a fabric config)")
    ring = topology == Topology.Ring
    avail = discovered_links(mesh, cluster_axis, ring)
    if num_links is None:
        return avail
    if num_links < 1 or num_links > avail:
        raise ValueError(
            f"distributed_rms_norm: num_links={num_links} outside [1, {avail}] (links discovered on every hop, "
            f"topology={topology}, fabric={ttnn.get_fabric_config()})"
        )
    return num_links


def _classify(input_tensor, cluster_axis, gamma, topology, num_links, cfg):
    has_gamma = gamma is not None
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "gamma_mode": "gamma" if has_gamma else "no_gamma",
        "gamma_dtype": gamma.dtype if has_gamma else "none",
        "gamma_layout": gamma.layout if has_gamma else "none",
        "memory_layout": input_tensor.memory_config().memory_layout,
        "cluster_axis": cluster_axis,
        "topology": topology,
        "num_links": num_links,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(_per_device_shape(input_tensor)),), axes)
    return axes


def validate(
    input_tensor,
    *,
    cluster_axis,
    gamma=None,
    topology=Topology.Linear,
    num_links=None,
    compute_kernel_config=None,
    memory_config=None,
):
    """Hard validation, then the support gate. Returns (resolved compute config, canonical num_links)."""
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    links = _hard_validate(input_tensor, cluster_axis, gamma, topology, num_links)
    axes = _classify(input_tensor, cluster_axis, gamma, topology, links, cfg)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"distributed_rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"distributed_rms_norm: unsupported combination (refinement candidate): {exc}")

    # Output placement: the input's own shard spec packs in place (zero-copy); any other output (interleaved, or a
    # different shard spec / an interleaved input with a sharded output: resharding) streams through the writer's
    # output TensorAccessor inside the same dispatch (Refinement 1).
    return cfg, links


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def distributed_rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    cluster_axis: int,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
    topology=Topology.Linear,
    num_links: Optional[int] = None,
    compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    cfg, links = validate(
        input_tensor,
        cluster_axis=cluster_axis,
        gamma=gamma,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )
    mesh = input_tensor.device()
    out_mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        mesh,
        out_mc,
    )
    mesh_desc, res = create_mesh_program_descriptor(
        input_tensor,
        output_tensor,
        gamma=gamma,
        epsilon=epsilon,
        cluster_axis=cluster_axis,
        ring=topology == Topology.Ring,
        num_links=links,
        compute_kernel_config=cfg,
    )
    io = [input_tensor] + ([gamma] if gamma is not None else []) + [res.landing, output_tensor]
    n_collected = collected_count()
    out = ttnn.generic_op(io, mesh_desc)
    if collected_count() != n_collected:
        forget_resources(res)  # dispatch was intercepted: the resources were never initialized on device
    return out


def plan_summary(
    input_tensor, *, cluster_axis, gamma=None, topology=Topology.Linear, num_links=None, compute_kernel_config=None
):
    """Host-only: the regime and split the op would run (for tests)."""
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    links = _hard_validate(input_tensor, cluster_axis, gamma, topology, num_links)
    return _plan_summary(
        input_tensor, cluster_axis=cluster_axis, num_links=links, gamma=gamma, fp32_dest_acc_en=cfg.fp32_dest_acc_en
    )
