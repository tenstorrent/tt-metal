# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Scaled-dot-product attention (fused on-device kernel).

Computes ``O = softmax((Q @ K^T) * scale + mask) @ V`` using a
Flash-Attention-1-style online softmax — memory is O(D) per query
tile-row, independent of S_kv.

See ``op_design.md`` for the full algorithm and CB contract.

SUPPORTED (post-Refinement-3):
    dtype          : bfloat16, float32, bfloat8_b
    layout         : TILE_LAYOUT
    alignment      : tile_aligned, w_non_aligned, h_non_aligned
    attention_kind : self, cross
    kv_heads_mode  : mha, mqa, gqa  (H_q % H_kv == 0; reader broadcasts KV heads)
    mask_mode      : none, causal
    scale_mode     : auto, explicit

EXCLUSIONS:
    {"mask_mode": "causal", "attention_kind": "cross"} — causal cut on
    a rectangular S_q × S_kv block is well-defined math but not a real
    workload; kept here so refinements have to consciously enable it.
    {"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"} +
    {"dtype": ttnn.bfloat8_b, "alignment": "h_non_aligned"} — bf8b's
    per-tile shared-exponent storage makes the in-kernel synthetic
    alignment-mask path lossy (the -inf bit pattern depends on the
    face's exponent, not just the element bytes). Deferred to its own
    refinement.
"""

from __future__ import annotations

import math
from typing import Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .scaled_dot_product_attention_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Project Q's (S_q, D) onto the 3-value alignment axis.

    Matches the contract documented in the op prompt + feature_spec:
      - tile_aligned   — both S_q and D divisible by 32.
      - w_non_aligned  — D not divisible by 32 (D is the "W" / inner dim).
      - h_non_aligned  — D aligned, S_q not aligned.

    A single "non_tile_aligned" bucket would collapse two distinct
    refinement targets — the W-side fix (head_dim padding, register
    masking) and the H-side fix (query/key seq-len padding) require
    different kernel changes, so the tagger surfaces them separately
    even though Phase 0 doesn't yet handle either.
    """
    q_shape = inputs[0]
    if len(q_shape) < 4:
        # No meaningful alignment for ill-formed shapes — collapse to
        # the W bucket so the case still falls outside "tile_aligned".
        return "w_non_aligned"
    s_q = q_shape[-2]
    d = q_shape[-1]
    d_ok = (d % 32) == 0
    s_ok = (s_q % 32) == 0
    if d_ok and s_ok:
        return "tile_aligned"
    if not d_ok:
        return "w_non_aligned"
    return "h_non_aligned"


def tag_attention_kind(inputs, axes):
    """self when S_q == S_kv, cross otherwise. Driven by Q vs K shapes."""
    q_shape = inputs[0]
    k_shape = inputs[1]
    return "self" if q_shape[-2] == k_shape[-2] else "cross"


def tag_kv_heads(inputs, axes):
    """mha when H_q == H_kv; mqa when H_kv == 1 < H_q; gqa otherwise.

    Drives the refinement queue's KV-head bucket: Phase 0 only handles
    MHA (the reader assumes H_kv == H_q in its KV-base computation), so
    GQA/MQA cells xfail until a refinement broadens the kernel.
    """
    q_shape = inputs[0]
    k_shape = inputs[1]
    h_q = q_shape[1]
    h_kv = k_shape[1]
    if h_q == h_kv:
        return "mha"
    if h_kv == 1:
        return "mqa"
    return "gqa"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "attention_kind": tag_attention_kind,
    "kv_heads_mode": tag_kv_heads,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    # Refinement 1: float32 + bfloat8_b added. The kernel's CB-format
    # plumbing keys off q.dtype for input/output CBs; running-state CBs
    # (cb_cur_max etc.) are Float32 when fp32_dest_acc_en=True (the
    # default), independent of input dtype. Refinement 5 reclaimed the
    # vestigial cb_prev_* CBs that R4-iter3 stopped using — see the
    # program descriptor's top-of-file note.
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
    # Refinement 3: alignment extended to {tile_aligned, w_non_aligned,
    # h_non_aligned}. Phase 0 only ran tile_aligned; the implementation
    # now relies on (a) TTNN's TILE_LAYOUT auto-zeroing the tile padding
    # — head_dim zero-pad is benign in the QK^T reduction (0*x=0) and
    # the attn@V projection (0 in V padded cols → 0 in output's padded
    # cols), and (b) a kernel-side synthetic -inf mask overlay over the
    # last K tile's padded key positions for S_kv non-aligned cases —
    # softmax sees exp(-inf)=0 and the padded keys drop out of both the
    # numerator and the denominator. The synthetic overlay composes with
    # an optional user mask in the reader (mask CB always exists when
    # either path is active).
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "attention_kind": ["self", "cross"],
    # Refinement 2: kv_heads_mode extended to {mha, mqa, gqa}. Reader
    # computes h_kv = h_q / (H_q / H_kv) (integer divide; H_q % H_kv == 0
    # per the op contract). mha is exactly H_q == H_kv (group_size=1, the
    # broadcast collapses to identity). mqa is H_kv == 1 (every Q-head
    # reads the single K/V head). gqa covers 1 < H_kv < H_q.
    "kv_heads_mode": ["mha", "mqa", "gqa"],
    "mask_mode": ["none", "causal"],
    "scale_mode": ["auto", "explicit"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # Causal mask on a rectangular S_q × S_kv block is well-defined math
    # but not a real workload. Refusing it keeps refinements honest about
    # what's intended vs. accidentally working.
    {"mask_mode": "causal", "attention_kind": "cross"},
    # Refinement 3: bf8b + non-aligned (W or H). The kernel's synthetic
    # alignment mask writes -inf bit patterns at known L1 positions in a
    # full tile, but bf8b stores a per-face shared exponent — the bit
    # pattern that *means* -inf depends on that face's exponent, not on
    # a fixed BF16/FP32 immediate. Refinement candidate: either (a)
    # pre-cook the alignment mask host-side in bf16 and reconfig the
    # mask CB dtype to bf16 regardless of q.dtype, or (b) tilize-wrap
    # the bf8b path so the synthetic mask lives in a separate bf16 CB.
    {"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"},
    {"dtype": ttnn.bfloat8_b, "alignment": "h_non_aligned"},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    attention_mask: Optional[ttnn.Tensor] = None,
    scale: Optional[float] = None,
) -> None:
    # ---- shape / rank / structural sanity --------------------------------
    if len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4:
        raise UnsupportedAxisValue(
            f"scaled_dot_product_attention: Q/K/V must be rank-4, got "
            f"Q={list(q.shape)} K={list(k.shape)} V={list(v.shape)}"
        )

    Bq, Hq, Sq, Dq = list(q.shape)
    Bk, Hk, Skv, Dk = list(k.shape)
    Bv, Hv, Skv_v, Dv = list(v.shape)

    if not (Bq == Bk == Bv):
        raise UnsupportedAxisValue(f"scaled_dot_product_attention: batch mismatch B={Bq}/{Bk}/{Bv}")
    if Hk != Hv:
        # K and V must share head-count (they index the same KV cache).
        # Q may differ (GQA / MQA) — the kv_heads_mode axis tracks that.
        raise UnsupportedAxisValue(f"scaled_dot_product_attention: K/V head count mismatch (Hk={Hk}, Hv={Hv})")
    # Refinement 2 (GQA / MQA): the reader's KV-head broadcast computes
    # group_size = H_q / H_kv and h_kv = h_q / group_size. The integer
    # division is only well-defined when H_q is a multiple of H_kv —
    # otherwise the same H_kv tile would be read by an uneven number of
    # Q heads, breaking the H_q-head output. Reject early with a clear
    # message rather than letting the kernel silently produce garbage
    # for the trailing Q heads.
    if Hq % Hk != 0:
        raise UnsupportedAxisValue(
            f"scaled_dot_product_attention: GQA/MQA requires H_q ({Hq}) be a "
            f"multiple of H_kv ({Hk}); got remainder {Hq % Hk}"
        )
    if not (Dq == Dk == Dv):
        raise UnsupportedAxisValue(f"scaled_dot_product_attention: D mismatch D={Dq}/{Dk}/{Dv}")
    if Skv != Skv_v:
        raise UnsupportedAxisValue(f"scaled_dot_product_attention: K/V S_kv mismatch ({Skv} vs {Skv_v})")

    inputs = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
    if attention_mask is not None:
        inputs = inputs + (tuple(attention_mask.shape),)

    axes = {
        "dtype": q.dtype,
        "layout": q.layout,
        "mask_mode": "causal" if attention_mask is not None else "none",
        "scale_mode": "auto" if scale is None else "explicit",
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    # ---- per-axis SUPPORTED check ----------------------------------------
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(
                f"scaled_dot_product_attention: {axis}={axes[axis]!r} " f"not in SUPPORTED {allowed}"
            )

    # ---- cell-level EXCLUSIONS check -------------------------------------
    for exc in EXCLUSIONS:
        if all(axes.get(k_) == v_ for k_, v_ in exc.items()):
            raise ExcludedCell(
                f"scaled_dot_product_attention: unsupported combination " f"(refinement candidate): {exc}"
            )

    # ---- K/V dtype/layout sanity (same as Q) -----------------------------
    for name, t in (("K", k), ("V", v)):
        if t.dtype != q.dtype:
            raise UnsupportedAxisValue(f"scaled_dot_product_attention: {name}.dtype={t.dtype} != Q.dtype={q.dtype}")
        if t.layout != q.layout:
            raise UnsupportedAxisValue(f"scaled_dot_product_attention: {name}.layout={t.layout} != Q.layout={q.layout}")

    # ---- mask sanity (if present) ----------------------------------------
    if attention_mask is not None:
        if len(attention_mask.shape) != 4:
            raise UnsupportedAxisValue(
                f"scaled_dot_product_attention: mask must be rank-4, got " f"{list(attention_mask.shape)}"
            )
        Bm, Hm, Sqm, Skvm = list(attention_mask.shape)
        if Bm != Bq:
            raise UnsupportedAxisValue(f"scaled_dot_product_attention: mask batch {Bm} != Q batch {Bq}")
        if Hm not in (1, Hq):
            raise UnsupportedAxisValue(f"scaled_dot_product_attention: mask head count {Hm} must be 1 or {Hq}")
        if Sqm != Sq or Skvm != Skv:
            raise UnsupportedAxisValue(
                f"scaled_dot_product_attention: mask S dims ({Sqm}, {Skvm}) " f"!= ({Sq}, {Skv})"
            )
        # Mask must match Q's dtype (the kernel's CB layout pairs mask
        # with q.dtype so copy_tile-and-add operates in the same
        # precision as the QK^T accumulator).
        if attention_mask.dtype != q.dtype or attention_mask.layout != ttnn.TILE_LAYOUT:
            raise UnsupportedAxisValue(
                f"scaled_dot_product_attention: mask must match Q's dtype ({q.dtype}) "
                f"+ TILE, got dtype={attention_mask.dtype} layout={attention_mask.layout}"
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# Default compute config — matches the pre-Refinement-1 hard-coded
# behavior (fp32_dest_acc_en=True with the platform-default HiFi4 math
# fidelity). Centralized so the entry point and any caller introspection
# point at the same defaults.
_DEFAULT_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    dst_full_sync_en=False,
)


def scaled_dot_product_attention(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    attention_mask: Optional[ttnn.Tensor] = None,
    scale: Optional[float] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    compute_kernel_config: Optional["ttnn.WormholeComputeKernelConfig"] = None,
) -> ttnn.Tensor:
    """Fused on-device scaled-dot-product attention.

    O = softmax((Q @ K^T) * scale + mask) @ V

    Shapes:
        Q : (B, H, S_q,  D)
        K : (B, H, S_kv, D)
        V : (B, H, S_kv, D)
        mask : (B, 1, S_q, S_kv) or (B, H, S_q, S_kv) — optional, additive
        Output : (B, H, S_q, D)

    Args:
        compute_kernel_config: optional ``ttnn.WormholeComputeKernelConfig``
            controlling math_fidelity, fp32_dest_acc_en, math_approx_mode,
            and dst_full_sync_en. When omitted, defaults to
            HiFi4 + fp32_dest_acc_en=True (matches pre-Refinement-1 behavior).
    """
    validate(q, k, v, attention_mask=attention_mask, scale=scale)

    # The fused-scale-exp SFPU path (`exp_tile<scale_en=true>`) used in
    # the compute kernel requires the destination to be in fp32
    # accumulation mode — without it the TTI_SFPMULI immediate-constraint
    # asm fails to build. Reject the bad config early with a clear
    # message so callers don't see an opaque kernel-build TT_THROW.
    if compute_kernel_config is not None and not bool(compute_kernel_config.fp32_dest_acc_en):
        raise UnsupportedAxisValue(
            "scaled_dot_product_attention: compute_kernel_config.fp32_dest_acc_en=False "
            "is not supported (the fused-scale-exp SFPU path requires fp32 DEST). "
            "Set fp32_dest_acc_en=True or pass compute_kernel_config=None to use defaults."
        )

    device = q.device()
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    B, H, Sq, _ = list(q.shape)
    _, _, _, D = list(q.shape)
    output_shape = [B, H, Sq, D]

    # Default scale: 1 / sqrt(D), matching the PyTorch reference.
    scale_value = float(scale) if scale is not None else 1.0 / math.sqrt(D)

    effective_config = compute_kernel_config if compute_kernel_config is not None else _DEFAULT_COMPUTE_KERNEL_CONFIG
    # WormholeComputeKernelConfig() defaults math_fidelity to Invalid;
    # canonicalize so the program descriptor never sees Invalid.
    if effective_config.math_fidelity == ttnn.MathFidelity.Invalid:
        effective_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=effective_config.math_approx_mode,
            fp32_dest_acc_en=effective_config.fp32_dest_acc_en,
            dst_full_sync_en=effective_config.dst_full_sync_en,
        )

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        q.dtype,
        q.layout,
        device,
        out_mem,
    )

    program_descriptor = create_program_descriptor(
        q,
        k,
        v,
        output_tensor,
        attention_mask=attention_mask,
        scale_value=scale_value,
        compute_kernel_config=effective_config,
    )

    # Output tensor MUST be last; optional mask, when present, goes before output.
    tensors = [q, k, v]
    if attention_mask is not None:
        tensors.append(attention_mask)
    tensors.append(output_tensor)

    return ttnn.generic_op(tensors, program_descriptor)
