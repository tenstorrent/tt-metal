# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""distributed_rms_norm — RMSNorm over a hidden dim sharded across one mesh axis, fused with the cross-device
exchange of the per-row Σx² partials over the fabric (op_design.md).

Every device p of a group of G devices along `cluster_axis` holds x[p] (..., M, W_local) and returns
    out[p] = x[p] * rsqrt(Σ_g Σ_w x[g]^2 / (G * W_local) + eps) * gamma[p].

ONE ttnn.generic_op dispatch per call (one MeshProgramDescriptor, one program per chip): the local half is the base
op rms_norm's blocking (R1 / R3-i / R1-HS / R3-s on the grid left after the port cores), the cross-device half is a
relay all-gather of one fp32 record (32 words) per tile-row and device, driven by 2L port cores per chip.

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() below drive both the runtime gate and the
golden-test xfail decisions.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology

from .distributed_rms_norm_program_descriptor import (
    build_groups,
    create_mesh_program_descriptor,
    usable_links,
)


# ---------------------------------------------------------------------------
# Default compute config — the ONE place the Phase 0 default is defined.
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """Fresh Phase 0 default compute config (maxed-out precision corner)."""
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
    """tile_aligned (H%32==0 and W%32==0) / w_non_aligned (W%32!=0) / h_non_aligned (only H misaligned)."""
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
    "num_links": [1, 2],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # Natively rejected precision combination (precision_convention.md).
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # Carried from the base op: the W-pad columns of a Bfp8_b tile cannot be zeroed element-wise (16 datums share
    # one exponent), so a non-aligned W cannot keep its tile padding out of Σx².
    {"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _hard_validate(input_tensor, cluster_axis, gamma, topology, num_links):
    """Caller errors (ValueError), raised before any support check. Returns (G, L_usable, ring)."""
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"distributed_rms_norm: input must have rank >= 2, got rank {len(shape)}")
    W = shape[-1]
    if gamma is not None:
        g_shape = list(gamma.shape)
        if gamma.layout == ttnn.TILE_LAYOUT:
            want = [1, 1, 1, W]
        else:
            want = [1, 1, -(-W // TILE_DIM), TILE_DIM]
        if g_shape != want:
            raise ValueError(
                f"distributed_rms_norm: gamma ({gamma.layout}) must have per-device shape {want}, got {g_shape}"
            )
    if cluster_axis not in (0, 1):
        raise ValueError(f"distributed_rms_norm: cluster_axis must be 0 or 1, got {cluster_axis}")
    mesh = input_tensor.device()
    mesh_shape = tuple(mesh.shape)
    G = mesh_shape[cluster_axis]
    if G < 2:
        raise ValueError(f"distributed_rms_norm: cluster_axis={cluster_axis} has size {G}; nothing to reduce")
    ring = topology == Topology.Ring and G >= 3
    if ring:
        cfg = ttnn.get_fabric_config()
        F = ttnn.FabricConfig
        wraps = {
            F.FABRIC_2D_TORUS_XY: (0, 1),
            F.FABRIC_2D_TORUS_X: (1,),
            F.FABRIC_2D_TORUS_Y: (0,),
        }
        for name in ("FABRIC_1D_RING",):
            if hasattr(F, name):
                wraps[getattr(F, name)] = (0, 1)
        if cluster_axis not in wraps.get(cfg, ()):
            raise ValueError(
                f"distributed_rms_norm: Ring along cluster_axis={cluster_axis} needs a fabric config that wraps it "
                f"(active: {cfg})"
            )
    L_usable = usable_links(mesh, build_groups(mesh_shape, cluster_axis), ring)
    if L_usable < 1:
        raise ValueError(
            f"distributed_rms_norm: some hop of the group has no forwarding link "
            f"(topology={topology}, cluster_axis={cluster_axis})"
        )
    if num_links is not None and (num_links <= 0 or num_links > L_usable):
        raise ValueError(f"distributed_rms_norm: num_links={num_links} outside [1, {L_usable}] usable links per hop")
    return G, L_usable, ring


def _classify(input_tensor, gamma, cfg, cluster_axis, topology, num_links):
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
        axes[axis_name] = tagger((list(input_tensor.shape),), axes)
    return axes


def _same_placement(out_mc, in_mc) -> bool:
    if out_mc.memory_layout != in_mc.memory_layout:
        return False
    if in_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return True
    a, b = out_mc.shard_spec, in_mc.shard_spec
    return (
        out_mc.buffer_type == in_mc.buffer_type
        and a is not None
        and b is not None
        and list(a.shape) == list(b.shape)
        and a.grid == b.grid
        and a.orientation == b.orientation
    )


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
    """Hard validation (ValueError), then the registry gate (UnsupportedAxisValue / ExcludedCell), then the regime
    refusals (NotImplementedError). Returns (cfg, G, L, ring)."""
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    G, L_usable, ring = _hard_validate(input_tensor, cluster_axis, gamma, topology, num_links)
    L = L_usable if num_links is None else num_links
    axes = _classify(input_tensor, gamma, cfg, cluster_axis, topology, L)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"distributed_rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    if memory_config is not None and memory_config.memory_layout not in SUPPORTED["memory_layout"]:
        raise UnsupportedAxisValue(
            f"distributed_rms_norm: output memory_layout={memory_config.memory_layout!r} not in SUPPORTED"
        )
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"distributed_rms_norm: unsupported combination (refinement candidate): {exc}")

    # Regime refusal (b): a sharded input writes its own shard spec; an interleaved input an interleaved output.
    in_mc = input_tensor.memory_config()
    if memory_config is not None and not _same_placement(memory_config, in_mc):
        if not (
            in_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
            and memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        ):
            raise NotImplementedError(
                f"distributed_rms_norm: output placement {memory_config} must match the input placement {in_mc}"
            )
    return cfg, G, L, ring


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
    cfg, G, L, ring = validate(
        input_tensor,
        cluster_axis=cluster_axis,
        gamma=gamma,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )
    mesh = input_tensor.device()
    in_mc = input_tensor.memory_config()
    if in_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    else:
        output_memory_config = memory_config if memory_config is not None else in_mc
    # allocate_tensor_on_device takes positional args only.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        mesh,
        output_memory_config,
    )
    mesh_desc, landing = create_mesh_program_descriptor(
        input_tensor,
        output_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=cfg,
        cluster_axis=cluster_axis,
        G=G,
        num_links=L,
        ring=ring,
    )
    io_tensors = [input_tensor] + ([gamma] if gamma is not None else []) + [landing, output_tensor]
    # Output tensor must be last.
    return ttnn.generic_op(io_tensors, mesh_desc)
