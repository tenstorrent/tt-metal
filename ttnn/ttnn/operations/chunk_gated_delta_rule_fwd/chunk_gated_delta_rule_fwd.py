# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""chunk_gated_delta_rule_fwd — training-side forward of the chunked gated delta rule.

One native TTNN device-program dispatch (`ttnn.generic_op`) returns the output `o`, the final
recurrent state and every intermediate a backward consumes: `h` (per-chunk ENTERING states),
`v_new`, `g_cumsum` and the UT inverse `A`.  The program is phased into three stages (P: item-parallel
prep, S: V-split sequential scan, E: item-parallel output assembly) sequenced by segmented
semaphore handoffs — see `op_design.md` and the program descriptor.

Registry-model op file: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate().  INVALID is absent on
purpose — it lives in the golden suite's feature_spec.py.

CALLER CONTRACT: `q` and `k` MUST be L2-normalized along K.  Without it the UT transform is not
contractive and the forward diverges; the op does not normalize.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .chunk_gated_delta_rule_fwd_program_descriptor import (
    MAX_HEADS,
    build_program,
    default_compute_kernel_config,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — over `inputs = ((B, T, H, K, V), chunk_size)`; chunk_size first so
#    seq_alignment can read it.
# ---------------------------------------------------------------------------


def tag_chunk_size(inputs, axes):
    return int(inputs[1])


def tag_seq_alignment(inputs, axes):
    t = int(inputs[0][1])
    return "chunk_aligned" if t % int(axes["chunk_size"]) == 0 else "chunk_ragged"


def tag_head_dims(inputs, axes):
    k, v = int(inputs[0][3]), int(inputs[0][4])
    return "square" if k == v else "wide_v"


INPUT_TAGGERS = {
    "chunk_size": tag_chunk_size,
    "seq_alignment": tag_seq_alignment,
    "head_dims": tag_head_dims,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "state_mode": ["no_h0", "with_h0"],
    "chunk_size": [32, 64],
    "seq_alignment": ["chunk_aligned", "chunk_ragged"],
    "head_dims": ["square", "wide_v"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
    "math_fidelity": {"value": ["LoFi", "HiFi2", "HiFi3", "HiFi4"], "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _shape(t):
    return [int(d) for d in t.shape]


def validate(q, k, v, g, beta, *, initial_state=None, chunk_size=64, compute_kernel_config=None, **_):
    qs, ks, vs, gs, bs = (_shape(t) for t in (q, k, v, g, beta))
    if len(qs) != 4 or len(ks) != 4 or len(vs) != 4:
        raise ValueError("chunk_gated_delta_rule_fwd: q, k must be [B,T,H,K] and v [B,T,H,V] (rank 4)")
    if len(gs) != 3 or len(bs) != 3:
        raise ValueError("chunk_gated_delta_rule_fwd: g and beta must be [B,T,H] (rank 3)")
    B, T, H, K = qs
    V = vs[3]
    if ks != qs:
        raise ValueError(f"chunk_gated_delta_rule_fwd: k shape {ks} != q shape {qs}")
    if vs[:3] != [B, T, H]:
        raise ValueError(f"chunk_gated_delta_rule_fwd: v shape {vs} does not match q's [B,T,H]={[B, T, H]}")
    if gs != [B, T, H] or bs != [B, T, H]:
        raise ValueError(f"chunk_gated_delta_rule_fwd: g {gs} / beta {bs} must be [B,T,H]={[B, T, H]}")
    if initial_state is not None:
        hs = _shape(initial_state)
        if len(hs) != 4:
            raise ValueError("chunk_gated_delta_rule_fwd: initial_state must be [B,H,K,V] (rank 4)")
        if hs != [B, H, K, V]:
            raise ValueError(f"chunk_gated_delta_rule_fwd: initial_state shape {hs} != {[B, H, K, V]}")
    if not isinstance(chunk_size, int) or chunk_size <= 0 or chunk_size % 32 != 0:
        raise ValueError(f"chunk_gated_delta_rule_fwd: chunk_size must be a positive multiple of 32, got {chunk_size}")
    if K % 32 != 0 or V % 32 != 0:
        raise ValueError(f"chunk_gated_delta_rule_fwd: K={K} and V={V} must be multiples of 32")

    # Every tensor must agree on dtype / layout (the axes below are read off q).
    others = [("k", k), ("v", v), ("g", g), ("beta", beta)]
    if initial_state is not None:
        others.append(("initial_state", initial_state))
    for name, t in others:
        if t.dtype != q.dtype:
            raise UnsupportedAxisValue(
                f"chunk_gated_delta_rule_fwd: mixed dtypes — {name}.dtype={t.dtype} != q.dtype={q.dtype}"
            )
        if t.layout != q.layout:
            raise UnsupportedAxisValue(
                f"chunk_gated_delta_rule_fwd: mixed layouts — {name}.layout={t.layout} != q.layout={q.layout}"
            )

    axes = {
        "dtype": q.dtype,
        "layout": q.layout,
        "state_mode": "no_h0" if initial_state is None else "with_h0",
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(((B, T, H, K, V), chunk_size), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"chunk_gated_delta_rule_fwd: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(key) == val for key, val in exc.items()):
            raise ExcludedCell(f"chunk_gated_delta_rule_fwd: unsupported combination (refinement candidate): {exc}")

    # Mechanism caps (hard errors, not support refusals).
    if H > MAX_HEADS:
        raise ValueError(
            f"chunk_gated_delta_rule_fwd: H={H} exceeds the mechanism cap of {MAX_HEADS} "
            "(the page index assumes ceil(H/32) == 1)"
        )
    if compute_kernel_config is not None and q.dtype == ttnn.float32 and not compute_kernel_config.fp32_dest_acc_en:
        raise ValueError("chunk_gated_delta_rule_fwd: float32 inputs require fp32_dest_acc_en=True")
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Debug hook: the last dispatch's internal scratch and geometry.  Not part of the public contract.
_LAST_DEBUG = {}


def chunk_gated_delta_rule_fwd(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    g: ttnn.Tensor,
    beta: ttnn.Tensor,
    *,
    initial_state: ttnn.Tensor = None,
    chunk_size: int = 64,
    scale: float = None,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = None,
):
    """Chunked gated delta rule forward.

    Args:
        q, k: `[B, T, H, K]`, L2-normalized along K (caller contract).
        v: `[B, T, H, V]`.
        g: `[B, T, H]` log-space decay gate (<= 0).
        beta: `[B, T, H]` write strength in (0, 1).
        initial_state: `[B, H, K, V]` or None (== zeros; never read).
        chunk_size: tokens per chunk, a positive multiple of 32.
        scale: query scale, default `K ** -0.5`.
        compute_kernel_config: `ttnn.ComputeConfigDescriptor`; None -> `default_compute_kernel_config()`.
        memory_config: output placement; defaults to q's.

    Returns:
        `(o, final_state, h, v_new, g_cumsum, A)`.
    """
    validate(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        chunk_size=chunk_size,
        compute_kernel_config=compute_kernel_config,
    )
    global _LAST_DEBUG
    outputs, debug = build_program(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        chunk_size=int(chunk_size),
        scale=scale,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )
    _LAST_DEBUG = debug
    return outputs


__all__ = [
    "chunk_gated_delta_rule_fwd",
    "default_compute_kernel_config",
    "validate",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
    "PROPERTIES",
]
