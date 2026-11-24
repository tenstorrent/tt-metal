# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""q_stem — the fused multi-device query stem of the DSA sparse MLA, as ONE ttnn.generic_op dispatch.

Per group of G devices along cluster_axis (group position p holds the column slice q_b[p] of q_b_proj):

    q[p]     = x @ q_b[p]                                    (B, 1, S, Hl * (Dn + Dr))
    heads[p] = q[p].view(B, S, Hl, Dn + Dr).permute(0, 2, 1, 3)
    device q' receives row quarter q' of every p (output head h = p * Hl + j)
    out[q']  = cat([nope[q'] @ wkv_b1, rope[q']], dim=-1)    (B, H, S/G, N + Dr)

Stage 1 (column-split matmul fed by an x multicast) keeps q in L1; port cores pull the remote quarters and send
them over the fabric straight into the destination cores' landing scratch; stage 3 (per-head matmul with wkv_b1 +
rope pass-through) runs as rows land. See op_design.md.
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology

# ---------------------------------------------------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """inputs = ((B, 1, S, Q), (1, 1, Q, Hl*(Dn+Dr)), (1, H, Dn, N), G)."""
    latent, _, _, G = inputs
    S = latent[-2]
    if S % 32 != 0 or S % G != 0 or (S // G) % 32 != 0:
        return "m_non_aligned"
    return "tile_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}

# ---------------------------------------------------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "q_b_proj_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "wkv_b1_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
    "output_layout": [ttnn.TILE_LAYOUT],
    "fp32_dest_acc_en": [True, False],
    "rope_mode": ["nope"],
    "alignment": ["tile_aligned"],
    "input_buffer": [ttnn.BufferType.DRAM, ttnn.BufferType.L1],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
    "cluster_axis": [1],
    "topology": [Topology.Ring],
    "num_links": [2],
}

# ---------------------------------------------------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------------------------------------------------
# Compute configuration (single source of the default)
# ---------------------------------------------------------------------------------------------------------------------


def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------------------------------------------------


def _shape(t):
    return tuple(int(v) for v in t.shape)


def _check_caller(input_tensor, q_b_proj_weight, wkv_b1_weight, cluster_axis, cos, sin, output_layout, num_links):
    """Caller errors (ValueError / RuntimeError), not support refusals."""
    for name, t in (
        ("input_tensor", input_tensor),
        ("q_b_proj_weight", q_b_proj_weight),
        ("wkv_b1_weight", wkv_b1_weight),
    ):
        if len(t.shape) != 4:
            raise ValueError(f"q_stem: {name} must be rank 4, got {list(t.shape)}")
    mesh = input_tensor.device()
    mesh_shape = tuple(mesh.shape)
    if cluster_axis not in (0, 1) or cluster_axis >= len(mesh_shape):
        raise ValueError(f"q_stem: cluster_axis {cluster_axis} out of range for mesh {mesh_shape}")
    G = mesh_shape[cluster_axis]
    if G < 2:
        raise ValueError(f"q_stem: cluster_axis {cluster_axis} has size {G}; nothing to exchange")
    B, one, S, Q = _shape(input_tensor)
    _, _, Qw, Cq = _shape(q_b_proj_weight)
    _, H, Dn, N = _shape(wkv_b1_weight)
    if one != 1:
        raise ValueError(f"q_stem: latent must be (B, 1, S, Q), got {list(input_tensor.shape)}")
    if Qw != Q:
        raise ValueError(f"q_stem: q_b_proj Q ({Qw}) differs from the latent's ({Q})")
    if H % G != 0:
        raise ValueError(f"q_stem: wkv_b1 head count {H} not divisible by G={G}")
    Hl = H // G
    if Cq % Hl != 0:
        raise ValueError(f"q_stem: q_b_proj width {Cq} not a multiple of Hl={Hl}")
    Dr = Cq // Hl - Dn
    if Dr <= 0:
        raise ValueError(f"q_stem: rope width Dr = {Dr} must be > 0")
    if S % G != 0:
        raise ValueError(f"q_stem: S={S} not divisible by G={G}")
    for v, nm in ((Q, "Q"), (Dn, "Dn"), (Dr, "Dr"), (N, "N")):
        if v % 32 != 0:
            raise ValueError(f"q_stem: {nm}={v} must be a multiple of 32")
    if (cos is None) != (sin is None):
        raise ValueError("q_stem: cos and sin go together")
    if cos is not None:
        for nm, t in (("cos", cos), ("sin", sin)):
            if _shape(t) != (1, 1, S // G, Dr):
                raise ValueError(f"q_stem: {nm} must be (1, 1, {S // G}, {Dr}), got {list(t.shape)}")
    if output_layout not in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
        raise ValueError(f"q_stem: output_layout must be TILE or ROW_MAJOR, got {output_layout}")
    if not isinstance(num_links, int) or num_links <= 0:
        raise ValueError(f"q_stem: num_links must be a positive int, got {num_links}")
    return G


def validate(
    input_tensor,
    q_b_proj_weight,
    wkv_b1_weight,
    *,
    cluster_axis,
    cos=None,
    sin=None,
    topology=Topology.Ring,
    num_links=2,
    output_layout=ttnn.TILE_LAYOUT,
    compute_kernel_config=None,
):
    G = _check_caller(input_tensor, q_b_proj_weight, wkv_b1_weight, cluster_axis, cos, sin, output_layout, num_links)
    cfg = compute_kernel_config or default_compute_kernel_config()
    mc = input_tensor.memory_config()
    axes = {
        "dtype": input_tensor.dtype,
        "q_b_proj_dtype": q_b_proj_weight.dtype,
        "wkv_b1_dtype": wkv_b1_weight.dtype,
        "layout": input_tensor.layout,
        "output_layout": output_layout,
        "fp32_dest_acc_en": bool(getattr(cfg, "fp32_dest_acc_en", True)),
        "rope_mode": "rope" if cos is not None else "nope",
        "input_buffer": mc.buffer_type,
        "memory_layout": mc.memory_layout,
        "cluster_axis": cluster_axis,
        "topology": topology,
        "num_links": num_links,
    }
    inputs = (_shape(input_tensor), _shape(q_b_proj_weight), _shape(wkv_b1_weight), G)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"q_stem: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"q_stem: unsupported combination (refinement candidate): {exc}")
    return axes


# ---------------------------------------------------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------------------------------------------------


def q_stem(
    input_tensor,
    q_b_proj_weight,
    wkv_b1_weight,
    *,
    cluster_axis,
    cos=None,
    sin=None,
    topology=Topology.Ring,
    num_links=2,
    output_layout=ttnn.TILE_LAYOUT,
    compute_kernel_config=None,
):
    validate(
        input_tensor,
        q_b_proj_weight,
        wkv_b1_weight,
        cluster_axis=cluster_axis,
        cos=cos,
        sin=sin,
        topology=topology,
        num_links=num_links,
        output_layout=output_layout,
        compute_kernel_config=compute_kernel_config,
    )
    from ttnn.operations.q_stem.q_stem_program_descriptor import run_q_stem

    cfg = compute_kernel_config or default_compute_kernel_config()
    return run_q_stem(
        input_tensor,
        q_b_proj_weight,
        wkv_b1_weight,
        cluster_axis=cluster_axis,
        num_links=num_links,
        compute_config=cfg,
    )
