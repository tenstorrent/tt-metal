# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_gather_rms_norm — fused all-gather + RMSNorm / LayerNorm over Fabric 2D, ONE generic_op per call.

Every device of a cluster_axis group holds its (..., M, W_local) slice; every member returns the full normalized
(..., M, G * W_local) row. Scheme (op_design.md): stats-first, gather-normalized — per-row partial stats go around
the group once, each chip normalizes only its own slice, and the port cores send the normalized slices straight
into the peers' final output pages.

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() below, then the public entry point.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology

from .all_gather_rms_norm_program_descriptor import (
    build_plan,
    create_mesh_program_descriptor,
    derive_geometry,
    usable_links,
)


# ---------------------------------------------------------------------------
# Phase 0 compute-config default — the ONE place the default lives.
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """Fresh Phase 0 config per call: HiFi4, fp32 DEST accumulation, exact SFPU math."""
    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = ttnn.MathFidelity.HiFi4
    config.fp32_dest_acc_en = True
    config.math_approx_mode = False
    return config


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS (per-device input shape)
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """W_local (-1) first: "w_non_aligned" whenever W_local % 32 != 0, else H (-2)."""
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


# gamma_dtype / gamma_layout are projected inside validate() (real value or the "none" sentinel), not registered as
# taggers: a tagger key stops TARGET iteration of that axis in the harness (base op's reason).
def tag_gamma_dtype(gamma):
    return gamma.dtype if gamma is not None else "none"


def tag_gamma_layout(gamma):
    return gamma.layout if gamma is not None else "none"


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    "fp32_dest_acc_en": [True],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, "none"],
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    "norm_type": ["rms_norm", "layer_norm"],
    "beta_mode": ["beta", "no_beta"],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
    "cluster_axis": [0, 1],
    "topology": [Topology.Linear, Topology.Ring],
    "num_links": [1, 2],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # Precision convention: float32 input requires fp32 DEST accumulation.
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # bfloat8_b with W_local % 32 != 0: the R_U element-shifted assembly (and the reader's W-tail zero-fill) address
    # datums by byte offset, which a 16-datum shared-exponent block cannot express. Measured (verifier, gate tier):
    # 19/19 cells severity=bug (PCC ~ 0) with this entry lifted. Base op reason, confirmed for this op.
    {"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"},
    # (bfloat8_b x h_non_aligned is NOT excluded: H padding is never element-masked here — rows are independent.
    #  Measured: 19/19 gate cells pass with the base op's exclusion lifted.)
]


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _mesh_shape(t):
    return tuple(t.device().shape)


def _volume(t):
    v = 1
    for d in t.shape:
        v *= int(d)
    return v


def _check_weight(name, w, W):
    shape = [int(s) for s in w.shape]
    if w.layout == ttnn.TILE_LAYOUT:
        want = [1, 1, 1, W]
    elif w.layout == ttnn.ROW_MAJOR_LAYOUT:
        want = [1, 1, -(-W // 32), 32]
    else:
        raise ValueError(f"all_gather_rms_norm: {name} layout {w.layout} is neither TILE nor ROW_MAJOR")
    if shape != want:
        raise ValueError(f"all_gather_rms_norm: {name} shape {shape} != {want} for W={W} ({w.layout})")


def validate(
    input_tensor,
    *,
    cluster_axis,
    gamma=None,
    beta=None,
    norm_type="rms_norm",
    epsilon=1e-6,
    topology=Topology.Linear,
    num_links=None,
    compute_kernel_config=None,
    memory_config=None,
    output_tensor=None,
):
    shape = [int(s) for s in input_tensor.shape]
    if len(shape) < 2:
        raise ValueError(f"all_gather_rms_norm: input rank must be >= 2, got {len(shape)}")
    if norm_type not in ("rms_norm", "layer_norm"):
        raise ValueError(f"all_gather_rms_norm: unknown norm_type {norm_type!r}")
    if beta is not None and gamma is None:
        raise ValueError("all_gather_rms_norm: beta requires gamma")
    if beta is not None and norm_type == "rms_norm":
        raise ValueError("all_gather_rms_norm: beta is only valid with norm_type='layer_norm'")
    if cluster_axis not in (0, 1):
        raise ValueError(f"all_gather_rms_norm: cluster_axis must be 0 or 1, got {cluster_axis}")
    if epsilon is None or not (float(epsilon) >= 0.0) or float(epsilon) == float("inf"):
        raise ValueError(f"all_gather_rms_norm: epsilon must be finite and >= 0, got {epsilon}")
    mesh_shape = _mesh_shape(input_tensor)
    G = mesh_shape[cluster_axis]
    if G < 2:
        raise ValueError(f"all_gather_rms_norm: cluster_axis {cluster_axis} has size {G}")
    W = G * shape[-1]
    if gamma is not None:
        _check_weight("gamma", gamma, W)
    if beta is not None:
        _check_weight("beta", beta, W)
        if beta.dtype != gamma.dtype or beta.layout != gamma.layout:
            raise ValueError("all_gather_rms_norm: beta must have gamma's dtype and layout")

    config = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "layout": input_tensor.layout,
        "gamma_mode": "gamma" if gamma is not None else "no_gamma",
        "gamma_dtype": tag_gamma_dtype(gamma),
        "gamma_layout": tag_gamma_layout(gamma),
        "norm_type": norm_type,
        "beta_mode": "beta" if beta is not None else "no_beta",
        "memory_layout": input_tensor.memory_config().memory_layout,
        "cluster_axis": cluster_axis,
        "topology": topology,
        "num_links": num_links,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)
    for axis, allowed in SUPPORTED.items():
        if axis == "num_links" and axes[axis] is None:
            continue  # None = all usable links, resolved against the live fabric below
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"all_gather_rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"all_gather_rms_norm: unsupported combination (refinement candidate): {exc}")

    # Phase 0 output placement: DRAM interleaved (the final-address landing walks DRAM banks).
    out_mc = output_tensor.memory_config() if output_tensor is not None else memory_config
    if out_mc is not None and (
        out_mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED or out_mc.buffer_type != ttnn.BufferType.DRAM
    ):
        raise UnsupportedAxisValue(f"all_gather_rms_norm: output memory config {out_mc} (Phase 0: DRAM interleaved)")
    if input_tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
        raise UnsupportedAxisValue("all_gather_rms_norm: Phase 0 input is DRAM interleaved")
    for name, w in (("gamma", gamma), ("beta", beta)):
        if w is not None and w.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise UnsupportedAxisValue(f"all_gather_rms_norm: {name} must be interleaved")
    if output_tensor is not None:
        want = shape[:-1] + [W]
        got = [int(s) for s in output_tensor.shape]
        if got != want or output_tensor.dtype != input_tensor.dtype or output_tensor.layout != input_tensor.layout:
            raise ValueError(
                f"all_gather_rms_norm: output_tensor must be {want} {input_tensor.dtype} {input_tensor.layout}, "
                f"got {got} {output_tensor.dtype} {output_tensor.layout}"
            )

    # Live-fabric requests (caller errors).
    mesh = input_tensor.device()
    ring = topology == Topology.Ring and G > 2
    links = usable_links(mesh, cluster_axis, ring)
    if not links:
        raise ValueError(
            f"all_gather_rms_norm: {topology} along cluster_axis {cluster_axis} needs a link on every hop "
            f"(fabric {ttnn.get_fabric_config()})"
        )
    if num_links is not None and (num_links < 1 or num_links > links):
        raise ValueError(f"all_gather_rms_norm: num_links={num_links}, usable links per hop: {links}")
    return links


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def all_gather_rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    cluster_axis: int,
    gamma: Optional[ttnn.Tensor] = None,
    beta: Optional[ttnn.Tensor] = None,
    norm_type: str = "rms_norm",
    epsilon: float = 1e-6,
    topology=Topology.Linear,
    num_links: Optional[int] = None,
    compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    output_tensor: Optional[ttnn.Tensor] = None,
) -> ttnn.Tensor:
    links = validate(
        input_tensor,
        cluster_axis=cluster_axis,
        gamma=gamma,
        beta=beta,
        norm_type=norm_type,
        epsilon=epsilon,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        output_tensor=output_tensor,
    )
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    if num_links is None:
        num_links = min(links, 2)
    mesh = input_tensor.device()
    G = tuple(mesh.shape)[cluster_axis]
    geo = derive_geometry(input_tensor.shape, G)
    plan = build_plan(mesh, cluster_axis, topology, num_links)
    if output_tensor is None:
        out_shape = [int(s) for s in input_tensor.shape][:-1] + [geo.W]
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(out_shape),
            input_tensor.dtype,
            ttnn.TILE_LAYOUT,
            mesh,
            memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        )
    desc, _persistent = create_mesh_program_descriptor(
        mesh,
        plan,
        geo,
        input_tensor,
        output_tensor,
        gamma,
        beta,
        is_ln=norm_type == "layer_norm",
        epsilon=float(epsilon),
        compute_kernel_config=compute_kernel_config,
    )
    io = [input_tensor] + [t for t in (gamma, beta) if t is not None] + [output_tensor]
    return ttnn.generic_op(io, desc)
