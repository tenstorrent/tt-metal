# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""distributed_rms_norm — fused multi-device RMSNorm over a hidden dim sharded along one mesh axis.

One ttnn.generic_op (MeshProgramDescriptor, one program per device) per call:
  workers   : read x once (resident in L1), sum x^2 per row of their W chunk, publish one fp32 record per
              (tile row, W chunk) to the port cores of their link; after the exchange, gather the G*S_w
              records of their rows, finalize rstd, normalize (x*gamma prescaled during the exchange stall).
  sharded   : (HEIGHT / WIDTH, L1) the worker set is the shard grid and each worker's block is its resident
              shard, read in place; WIDTH pre-combines the W-chunk records on chip (R5) before the exchange.
  ports     : per (link, direction), relay the per-device record slots over Fabric (ready fence, arrival
              counters, one-hop relay), then multicast a stats-ready increment to the worker rectangle.
See op_design.md for the blocking model.
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology

from .distributed_rms_norm_program_descriptor import (
    build_plan,
    caller_link_checks,
    run_distributed_rms_norm,
)


def default_compute_kernel_config():
    """Phase 0 default: the maxed-out precision corner (fresh object on every call)."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS (per-device shape inputs[0])
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
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
    "dtype": [ttnn.float32, ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout": [ttnn.TILE_LAYOUT],
    # h_non_aligned: rows are independent, so tile-padding rows only ever reach padding rows of the output.
    "alignment": ["tile_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, "none"],
    # *_SHARDED (L1): native zero-copy — the shard is the worker's block (cb_x_resident backed on it) and the
    # output packs straight into its shard when the output shard spec matches (op_design.md R9 / R10).
    "memory_layout": [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
    "cluster_axis": [0, 1],
    # Ring: generic in G via _schedule; verified on the snake-ring mock (1xN / Nx1), G = 4, 1 and 2 links.
    "topology": [Topology.Linear, Topology.Ring],
    # None = auto (all usable links on every hop): always legal, listed so the registry matches the op.
    "num_links": [1, 2, None],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # precision convention: float32 input requires fp32 accumulation (kept for the fp32_dest_acc_en=False refinement)
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _caller_errors(input_tensor, cluster_axis, gamma, topology, num_links):
    """ValueErrors for requests the call itself cannot satisfy (raised before the support gate)."""
    shape = tuple(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"distributed_rms_norm: input rank must be >= 2, got shape {shape}")
    if cluster_axis not in (0, 1):
        raise ValueError(f"distributed_rms_norm: cluster_axis must be 0 or 1, got {cluster_axis}")
    mesh_device = input_tensor.device()
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[cluster_axis] < 2:
        raise ValueError(f"distributed_rms_norm: mesh {mesh_shape} has size 1 along cluster_axis {cluster_axis}")
    if gamma is not None:
        if gamma.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(
                f"distributed_rms_norm: gamma must be ROW_MAJOR (one stick per device), got {gamma.layout}"
            )
        gshape = tuple(gamma.shape)
        if gshape[-1] != shape[-1]:
            raise ValueError(
                f"distributed_rms_norm: gamma last dim {gshape[-1]} != input last dim (W_local) {shape[-1]}"
            )
    mem = input_tensor.memory_config()
    if mem.is_sharded() and mem.buffer_type != ttnn.BufferType.L1:
        raise ValueError(f"distributed_rms_norm: a sharded input must live in L1, got {mem.buffer_type}")
    if num_links is not None and num_links <= 0:
        raise ValueError(f"distributed_rms_norm: num_links must be > 0, got {num_links}")
    caller_link_checks(mesh_device, cluster_axis, topology, num_links)


def _axes(input_tensor, *, cluster_axis, gamma, topology, num_links, compute_kernel_config):
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "gamma_mode": "gamma" if gamma is not None else "no_gamma",
        "gamma_dtype": gamma.dtype if gamma is not None else "none",
        "memory_layout": input_tensor.memory_config().memory_layout,
        "cluster_axis": cluster_axis,
        "topology": topology,
        "num_links": num_links,
    }
    for name, tagger in INPUT_TAGGERS.items():
        axes[name] = tagger((tuple(input_tensor.shape),), axes)
    return axes


def validate(
    input_tensor,
    *,
    cluster_axis,
    gamma=None,
    topology=Topology.Linear,
    num_links=None,
    compute_kernel_config=None,
    **_,
):
    _caller_errors(input_tensor, cluster_axis, gamma, topology, num_links)
    axes = _axes(
        input_tensor,
        cluster_axis=cluster_axis,
        gamma=gamma,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
    )
    for axis, allowed in SUPPORTED.items():
        value = axes[axis]
        if value not in allowed:
            raise UnsupportedAxisValue(f"distributed_rms_norm: {axis}={value!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"distributed_rms_norm: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def distributed_rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    cluster_axis: int,
    gamma: ttnn.Tensor | None = None,
    epsilon: float = 1e-6,
    topology=Topology.Linear,
    num_links: int | None = None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    validate(
        input_tensor,
        cluster_axis=cluster_axis,
        gamma=gamma,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
    )
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    plan = build_plan(input_tensor, cluster_axis=cluster_axis, topology=topology, num_links=num_links)
    return run_distributed_rms_norm(
        input_tensor,
        gamma,
        plan,
        epsilon=float(epsilon),
        compute_kernel_config=cfg,
        # default: a sharded input keeps its shard spec (native zero-copy output); interleaved -> DRAM
        memory_config=memory_config
        if memory_config is not None
        else (input_tensor.memory_config() if input_tensor.memory_config().is_sharded() else ttnn.DRAM_MEMORY_CONFIG),
    )
