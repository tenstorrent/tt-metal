# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ring_joint_scaled_dot_product_attention — sequence-parallel ring attention fused with a fabric all-gather of K/V.

One ttnn.generic_op over a MeshProgramDescriptor per call (design: op_design.md, regime R1
`qstationary_dram_kv`). Every device keeps its queries; its K/V head slices travel the group over the fabric into
persistent DRAM landing buffers (TRANSPORT, on dedicated port cores), while the compute cores run Q-stationary flash
attention over local -> joint -> remote key blocks, gating each remote segment on the landing counters (OVERLAP).
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs[0..2] = per-device Q / K / V shapes, inputs[3] (optional) = LOGICAL joint Q shape
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    q, k, v = inputs[0], inputs[1], inputs[2]
    dims = (q[-2], k[-2], q[-1], v[-1])
    return "tile_aligned" if all(d % 32 == 0 for d in dims) else "non_aligned"


def tag_attention_kind(inputs, axes):
    return "self" if inputs[0][-2] == inputs[1][-2] else "cross"


def tag_kv_heads_mode(inputs, axes):
    hq, hk, hv = inputs[0][1], inputs[1][1], inputs[2][1]
    if hq == hk == hv:
        return "mha"
    if hk == hv and 1 < hk < hq:
        return "gqa"
    if hk == hv == 1:
        return "mqa"
    if hk == 1 and hv == hq:
        return "mla"
    return "other"


def tag_joint_len(inputs, axes):
    if len(inputs) < 4 or inputs[3] is None:
        return "none"
    return "aligned" if inputs[3][-2] % 32 == 0 else "non_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "attention_kind": tag_attention_kind,
    "kv_heads_mode": tag_kv_heads_mode,
    "joint_len": tag_joint_len,
}

# ---------------------------------------------------------------------------
# 2. SUPPORTED (Phase 0)
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "kv_dtype": [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "attention_kind": ["self"],
    "kv_heads_mode": ["mha"],
    "joint_len": ["none", "aligned"],
    "joint_mode": ["none", "replicated"],
    "mask_mode": ["none"],
    "is_balanced": [False],
    "seq_padding": ["exact", "padded"],
    "cluster_axis": [0, 1],
    "topology": [Topology.Linear, Topology.Ring],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """The op's default compute config: HiFi2, fp32 DEST accumulation."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _axes_of_call(
    q, k, v, jq, *, logical_n, logical_l, cluster_axis, topology, is_causal, is_balanced, compute_kernel_config
):
    G = tuple(q.device().shape)[cluster_axis]
    if jq is None:
        joint_mode = "none"
    else:
        joint_mode = "sharded" if logical_l > 0 else "replicated"
    cfg = compute_kernel_config or default_compute_kernel_config()
    axes = {
        "dtype": q.dtype,
        "kv_dtype": k.dtype,
        "fp32_dest_acc_en": bool(getattr(cfg, "fp32_dest_acc_en", True)),
        "layout": q.layout,
        "joint_mode": joint_mode,
        "mask_mode": "causal" if is_causal else "none",
        "is_balanced": bool(is_balanced),
        "seq_padding": "padded" if logical_n < G * k.shape[-2] else "exact",
        "cluster_axis": cluster_axis,
        "topology": topology,
    }
    inputs = [tuple(q.shape), tuple(k.shape), tuple(v.shape)]
    if jq is not None:
        b, h, rows, d = tuple(jq.shape)
        inputs.append((b, h, logical_l if joint_mode == "sharded" else rows, d))
    for name, tagger in INPUT_TAGGERS.items():
        axes[name] = tagger(tuple(inputs), axes)
    return axes


def validate(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    joint_tensor_q=None,
    *,
    logical_n,
    logical_l=0,
    cluster_axis,
    topology=Topology.Linear,
    is_causal=False,
    is_balanced=False,
    compute_kernel_config=None,
):
    if cluster_axis not in (0, 1):
        raise UnsupportedAxisValue(f"ring_joint_sdpa: cluster_axis={cluster_axis!r} not in SUPPORTED [0, 1]")
    axes = _axes_of_call(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        joint_tensor_q,
        logical_n=logical_n,
        logical_l=logical_l,
        cluster_axis=cluster_axis,
        topology=topology,
        is_causal=is_causal,
        is_balanced=is_balanced,
        compute_kernel_config=compute_kernel_config,
    )
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"ring_joint_sdpa: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"ring_joint_sdpa: unsupported combination (refinement candidate): {exc}")
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def ring_joint_scaled_dot_product_attention(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    joint_tensor_q=None,
    joint_tensor_k=None,
    joint_tensor_v=None,
    *,
    logical_n,
    logical_l=0,
    cluster_axis,
    topology=Topology.Linear,
    num_links=None,
    is_causal=False,
    is_balanced=False,
    scale=None,
    compute_kernel_config=None,
    memory_config=None,
):
    validate(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        joint_tensor_q,
        logical_n=logical_n,
        logical_l=logical_l,
        cluster_axis=cluster_axis,
        topology=topology,
        is_causal=is_causal,
        is_balanced=is_balanced,
        compute_kernel_config=compute_kernel_config,
    )
    from .ring_joint_scaled_dot_product_attention_program_descriptor import run_ring_joint_sdpa

    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    return run_ring_joint_sdpa(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        joint_tensor_q,
        joint_tensor_k,
        joint_tensor_v,
        logical_n=logical_n,
        logical_l=logical_l,
        cluster_axis=cluster_axis,
        topology=topology,
        num_links=num_links,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )
