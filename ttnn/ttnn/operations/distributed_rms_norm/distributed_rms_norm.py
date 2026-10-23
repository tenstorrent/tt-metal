# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""distributed_rms_norm — RMSNorm over a hidden dim split across the G devices of a mesh group.

    out[p] = x[p] / sqrt(Σ_g Σ_w x[g]² / (G·W_local) + ε) · γ[p]

One native dispatch per call: one `MeshProgramDescriptor` (one `ProgramDescriptor` per chip) run by a single
`ttnn.generic_op` (op_design.md). Local work follows the base op's regimes (R1-X row-parallel, R3-X W-slice
row-groups with an in-chip root combine); the only cross-device data is each device's per-row partial Σx² (fp32),
all-gathered over Fabric by dedicated port cores with a one-hop store-and-forward relay.

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() drive the runtime gate and the golden xfails.
"""

from __future__ import annotations

from typing import Optional

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology

from . import distributed_rms_norm_program_descriptor as _pd

TILE_DIM = _pd.TILE_DIM


# ---------------------------------------------------------------------------
# Default compute config — the ONE place the Phase 0 default is defined.
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """Fresh Phase 0 default (maxed-out precision corner)."""
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
# 2. SUPPORTED (Phase 0)
# ---------------------------------------------------------------------------

SUPPORTED = {
    # bfloat8_b: cb_x_tiles / cb_output carry Bfp8_b pages; intermediates follow DEST (program descriptor dest_format).
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    # False = 16-bit DEST: cb_sumsq_partial / cb_x_gamma Float16_b; the stats pages stay Float32 (fp32 wire / sum).
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT],
    # W-pad columns of a row's last tile are zeroed by the reader before Σx² (interleaved and sharded); pad rows are
    # computed independently and land in the output's tile padding.
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, "none"],
    # ROW_MAJOR: per-device (1, 1, ceil(W_local/32), 32) sticks, stick r = row 0 of γ tile r (read natively by the writer).
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    # *_SHARDED: native, zero-copy (cb_x_tiles / cb_output backed on the input / output shard). HEIGHT -> R1-X over the
    # shard grid; WIDTH -> one R3-X row-group over the shard grid.
    "memory_layout": [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
    "cluster_axis": [0, 1],
    "topology": [Topology.Linear, Topology.Ring],
    # None = "every usable link" (resolved to L_avail on the live cluster); an explicit value is checked against the
    # links discovered on every hop as a caller error before this gate.
    "num_links": [None, 1, 2],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # fp32 input needs fp32 accumulation (precision_convention.md).
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _caller_errors(input_tensor, cluster_axis, gamma, num_links, topology):
    """Caller errors (ValueError), raised before the support gate. Returns the group / link facts."""
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"distributed_rms_norm: input rank must be >= 2, got {len(shape)}")
    if cluster_axis not in (0, 1):
        raise ValueError(f"distributed_rms_norm: cluster_axis must be 0 or 1, got {cluster_axis!r}")
    mesh = input_tensor.device()
    mesh_shape = tuple(mesh.shape)
    if mesh_shape[cluster_axis] < 2:
        raise ValueError(
            f"distributed_rms_norm: cluster_axis={cluster_axis} has size {mesh_shape[cluster_axis]} on mesh "
            f"{mesh_shape}; a group needs >= 2 devices"
        )
    if gamma is not None:
        W = shape[-1]
        g_shape = list(gamma.shape)
        if gamma.layout == ttnn.TILE_LAYOUT:
            want = [1, 1, 1, W]
        else:
            want = [1, 1, (W + TILE_DIM - 1) // TILE_DIM, TILE_DIM]
        if g_shape != want:
            raise ValueError(
                f"distributed_rms_norm: gamma ({gamma.layout}) must have per-device shape {want}, got {g_shape}"
            )
    if num_links is not None and num_links <= 0:
        raise ValueError(f"distributed_rms_norm: num_links must be > 0, got {num_links}")
    ring = topology == Topology.Ring and mesh_shape[cluster_axis] > 2
    if ring:
        _pd.check_ring_wrap(mesh, cluster_axis)
    l_avail = _pd.links_available(mesh, cluster_axis, ring)
    if l_avail < 1:
        raise ValueError(f"distributed_rms_norm: some hop of the group along cluster_axis={cluster_axis} has no link")
    if num_links is not None and num_links > l_avail:
        raise ValueError(
            f"distributed_rms_norm: num_links={num_links} > {l_avail} link(s) available on every hop of the group"
        )
    return l_avail


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
    memory_config=None,
):
    """Caller errors (ValueError), then the support gate (SUPPORTED per axis, then EXCLUSIONS).
    Returns (resolved compute config, L_avail)."""
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    l_avail = _caller_errors(input_tensor, cluster_axis, gamma, num_links, topology)
    axes = _classify(input_tensor, cluster_axis, gamma, topology, num_links, cfg)
    for axis, allowed in SUPPORTED.items():
        if axis in ("gamma_dtype", "gamma_layout") and axes[axis] == "none":
            continue  # "no weight" is always legal
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"distributed_rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"distributed_rms_norm: unsupported combination (refinement candidate): {exc}")
    return cfg, l_avail


def _output_memory_config(input_tensor, memory_config):
    """The caller's placement; default: a sharded input's own shard placement (compute packs into the output shard,
    zero-copy), else DRAM interleaved. Any other placement is written by the writer through a TensorAccessor."""
    if memory_config is not None:
        return memory_config
    in_mc = input_tensor.memory_config()
    return ttnn.DRAM_MEMORY_CONFIG if in_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED else in_mc


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
    cfg, l_avail = validate(
        input_tensor,
        cluster_axis=cluster_axis,
        gamma=gamma,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )
    mesh = input_tensor.device()
    out_mc = _output_memory_config(input_tensor, memory_config)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)), input_tensor.dtype, input_tensor.layout, mesh, out_mc
    )
    mesh_desc = _pd.create_mesh_program_descriptor(
        input_tensor,
        output,
        gamma=gamma,
        epsilon=epsilon,
        cluster_axis=cluster_axis,
        topology=topology,
        num_links=num_links if num_links is not None else l_avail,
        compute_kernel_config=cfg,
    )
    io = [input_tensor] + ([gamma] if gamma is not None else []) + [output]
    return ttnn.generic_op(io, mesh_desc)
