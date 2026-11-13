# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_gather_rms_norm — fused Fabric all-gather of the hidden dim + RMSNorm over the gathered row, as ONE native
ttnn.generic_op dispatch (one MeshProgramDescriptor) per call.

    full[q] = concat_p x[p]                         (..., M, W),  W = G * W_local
    out[q]  = full * rsqrt(mean_w(full^2) + eps) * gamma      on every member q of the cluster_axis group

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() below, then the public entry point.
op_design.md is the specification; all_gather_rms_norm_program_descriptor.py holds the plan and the program.
"""

from __future__ import annotations

from typing import Optional

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology

from .all_gather_rms_norm_program_descriptor import create_mesh_program_descriptor, discovered_links


# ---------------------------------------------------------------------------
# Phase 0 compute-config default — the ONE place the default lives.
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """Fresh config per call: HiFi4, fp32 DEST accumulation, exact SFPU math."""
    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = ttnn.MathFidelity.HiFi4
    config.fp32_dest_acc_en = True
    config.math_approx_mode = False
    return config


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS (per-device INPUT shape inputs[0])
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    shape = inputs[0]
    if shape[-1] % 32 != 0:
        return "w_non_aligned"
    if shape[-2] % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return int(len(inputs[0]))


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


def _gamma_dtype(gamma):
    return gamma.dtype if gamma is not None else "none"


def _gamma_layout(gamma):
    return gamma.layout if gamma is not None else "none"


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
    # Precision convention: fp32 input requires fp32 DEST accumulation.
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # (bfloat8_b x w_non_aligned is supported since Refinement 2: regime N composes from a bf16 staging row.)
]


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _volume(shape):
    v = 1
    for d in shape:
        v *= int(d)
    return v


def _axis_wraps(cluster_axis):
    cfg = ttnn.get_fabric_config()
    wraps = {
        ttnn.FabricConfig.FABRIC_2D_TORUS_X: (1,),
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y: (0,),
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY: (0, 1),
    }
    for name in ("FABRIC_1D_RING",):
        if hasattr(ttnn.FabricConfig, name):
            wraps[getattr(ttnn.FabricConfig, name)] = (0, 1)
    return cluster_axis in wraps.get(cfg, ())


def validate(
    input_tensor,
    *,
    cluster_axis,
    gamma=None,
    epsilon=1e-6,
    topology=Topology.Linear,
    num_links=None,
    compute_kernel_config=None,
    memory_config=None,
    output_tensor=None,
):
    shape = [int(d) for d in input_tensor.shape]
    if len(shape) < 2:
        raise ValueError(f"all_gather_rms_norm: input rank must be >= 2, got {len(shape)}")
    if epsilon is None or not (float(epsilon) >= 0.0) or float(epsilon) == float("inf"):
        raise ValueError(f"all_gather_rms_norm: epsilon must be finite and >= 0, got {epsilon}")
    if cluster_axis not in (0, 1):
        raise UnsupportedAxisValue(f"all_gather_rms_norm: cluster_axis={cluster_axis!r} not in SUPPORTED [0, 1]")
    mesh_device = input_tensor.device()
    mesh_shape = tuple(mesh_device.shape)
    G = mesh_shape[cluster_axis]
    if G < 2:
        raise ValueError(f"all_gather_rms_norm: cluster_axis={cluster_axis} has size {G}; nothing to gather")
    W = G * shape[-1]
    if gamma is not None:
        gshape = [int(d) for d in gamma.shape]
        if gamma.layout == ttnn.TILE_LAYOUT:
            ok = gshape == [1, 1, 1, W]
        else:
            ok = gshape == [1, 1, -(-W // 32), 32]
        if not ok:
            raise ValueError(
                f"all_gather_rms_norm: gamma shape {gshape} does not match its {gamma.layout} format for W={W}"
            )
    config = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()

    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "layout": input_tensor.layout,
        "gamma_mode": "gamma" if gamma is not None else "no_gamma",
        "gamma_dtype": _gamma_dtype(gamma),
        "gamma_layout": _gamma_layout(gamma),
        "memory_layout": input_tensor.memory_config().memory_layout,
        "cluster_axis": cluster_axis,
        "topology": topology,
    }
    if num_links is not None:
        axes["num_links"] = num_links
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axis in axes and axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"all_gather_rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"all_gather_rms_norm: unsupported combination (refinement candidate): {exc}")

    in_mc = input_tensor.memory_config()
    if in_mc.is_sharded() and in_mc.buffer_type != ttnn.BufferType.L1:
        raise UnsupportedAxisValue(f"all_gather_rms_norm: a sharded input must be L1-resident, got {in_mc.buffer_type}")
    out_mc = output_tensor.memory_config() if output_tensor is not None else memory_config
    if out_mc is not None:
        if out_mc.is_sharded() and out_mc.buffer_type != ttnn.BufferType.L1:
            raise UnsupportedAxisValue("all_gather_rms_norm: a sharded output must be L1-resident")
        if out_mc.memory_layout not in SUPPORTED["memory_layout"]:
            raise UnsupportedAxisValue(f"all_gather_rms_norm: output memory_layout={out_mc.memory_layout!r}")
    if gamma is not None and gamma.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise UnsupportedAxisValue("all_gather_rms_norm: gamma must be interleaved")

    # topology / links the live cluster can satisfy
    ring = topology == Topology.Ring and G > 2
    if topology == Topology.Ring and G > 2:
        if not _axis_wraps(cluster_axis):
            raise ValueError(
                f"all_gather_rms_norm: Ring needs a fabric config that wraps cluster_axis={cluster_axis} "
                f"(active: {ttnn.get_fabric_config()})"
            )
    found = discovered_links(mesh_device, cluster_axis, ring)
    if found == 0:
        raise ValueError(f"all_gather_rms_norm: no usable fabric link along cluster_axis={cluster_axis} ({topology})")
    if num_links is not None:
        if int(num_links) <= 0:
            raise ValueError(f"all_gather_rms_norm: num_links must be > 0, got {num_links}")
        if int(num_links) > found:
            raise ValueError(f"all_gather_rms_norm: num_links={num_links} > {found} usable link(s) on some hop")

    if output_tensor is not None:
        want = [*shape[:-1], W]
        if (
            [int(d) for d in output_tensor.shape] != want
            or output_tensor.dtype != input_tensor.dtype
            or output_tensor.layout != ttnn.TILE_LAYOUT
        ):
            raise ValueError(
                f"all_gather_rms_norm: output_tensor must be {want} {input_tensor.dtype} TILE, got "
                f"{list(output_tensor.shape)} {output_tensor.dtype} {output_tensor.layout}"
            )
    return found


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def all_gather_rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    cluster_axis: int,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
    topology=Topology.Linear,
    num_links: Optional[int] = None,
    compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    output_tensor: Optional[ttnn.Tensor] = None,
) -> ttnn.Tensor:
    found = validate(
        input_tensor,
        cluster_axis=cluster_axis,
        gamma=gamma,
        epsilon=epsilon,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        output_tensor=output_tensor,
    )
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    mesh_device = input_tensor.device()
    G = tuple(mesh_device.shape)[cluster_axis]
    ring = topology == Topology.Ring and G > 2
    L = int(num_links) if num_links is not None else found

    if output_tensor is None:
        shape = [int(d) for d in input_tensor.shape]
        out_mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([*shape[:-1], G * shape[-1]]),
            input_tensor.dtype,
            ttnn.TILE_LAYOUT,
            mesh_device,
            out_mc,
        )
    mesh_desc = create_mesh_program_descriptor(
        input_tensor,
        gamma,
        output_tensor,
        cluster_axis=cluster_axis,
        ring=ring,
        L=L,
        epsilon=float(epsilon),
        compute_kernel_config=compute_kernel_config,
    )
    io = [input_tensor] + ([gamma] if gamma is not None else []) + [output_tensor]
    ttnn.generic_op(io, mesh_desc)
    return output_tensor
