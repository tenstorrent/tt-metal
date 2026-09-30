# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_pre — the pre-sublayer half of one Manifold-Constrained Hyper-Connection (DeepSeek-V4), fused.

Per token row x (length n*C) of X (..., T, n*C):
    r     = rsqrt(mean(x^2) + norm_eps)
    mixes = (x @ W) * r
    pre   = sigmoid(a_pre  * mixes[0:n]  + b[0:n]) + eps
    post  = 2 * sigmoid(a_post * mixes[n:2n] + b[n:2n])
    comb  = Sinkhorn(a_res * mixes[2n:] + b[2n:])        (n x n, DeepSeek-V4 order)
    y     = sum_i pre[i] * x[i*C:(i+1)*C]
Returns (y, post, comb). One ttnn.generic_op dispatch per call.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .mhc_pre_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs = (X_shape, W_shape)
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """C (and so n*C) is a multiple of 32 by contract: only the token dim T (dim -2) can break alignment."""
    x_shape = inputs[0]
    return "tile_aligned" if x_shape[-2] % 32 == 0 else "h_non_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

# weight_dtype bfloat16: added by the verifier on measured evidence (every golden cell with a bf16 W passes;
# the CB formats / page sizes already follow the tensor dtypes via `_cb_table`, helpers reconfigure per operand).
# dtype bfloat16 (streams): Refinement 1. An fp32 W is split once per kernel on the SFPU into the exact bf16
# pair (W_hi, W_lo) and both products accumulate in one DEST window, so bf16-X x fp32-W post/comb went from
# rel-RMS 5.0-5.5e-4 (FPU reading W as ~tf32) to 2.4-3.1e-4 (the FPU in-tile accumulation floor).
SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "weight_dtype": [ttnn.float32, ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "alignment": ["tile_aligned", "h_non_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "math_fidelity": {"value": ["HiFi4"], "source": "declared"},
}


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """HiFi4, fp32 DEST, approx off."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _derive_n(mix):
    for n in range(1, 64):
        if n * (n + 2) == mix:
            return n
        if n * (n + 2) > mix:
            break
    raise ValueError(f"mhc_pre: proj_weight last dim {mix} is not n*(n+2) for any integer n")


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, proj_weight, proj_bias, *, compute_kernel_config=None, sinkhorn_iters=20):
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    x_shape = tuple(input_tensor.shape)
    w_shape = tuple(proj_weight.shape)
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "weight_dtype": proj_weight.dtype,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((x_shape, w_shape), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"mhc_pre: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"mhc_pre: unsupported combination (refinement candidate): {exc}")

    # Structural contract.
    if len(x_shape) < 2 or len(x_shape) > 4:
        raise ValueError(f"mhc_pre: input rank must be 2..4, got {len(x_shape)}")
    if len(w_shape) != 2:
        raise ValueError(f"mhc_pre: proj_weight must be 2D, got {w_shape}")
    n = _derive_n(w_shape[-1])
    if n * (n + 2) + 1 > 32:
        raise ValueError(f"mhc_pre: n={n} needs {n * (n + 2) + 1} coefficient slots (> 32)")
    nc = x_shape[-1]
    if nc % n != 0 or (nc // n) % 32 != 0:
        raise ValueError(f"mhc_pre: last dim {nc} must be n*C with C % 32 == 0 (n={n})")
    if w_shape[0] != nc:
        raise ValueError(f"mhc_pre: proj_weight shape {w_shape} does not match input last dim {nc}")
    b_shape = tuple(proj_bias.shape)
    if b_shape[-1] != n * (n + 2) or proj_bias.dtype != ttnn.float32 or proj_bias.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"mhc_pre: proj_bias must be float32 TILE of shape (1, {n * (n + 2)}), got {b_shape}")
    if proj_weight.layout != ttnn.TILE_LAYOUT:
        raise ValueError("mhc_pre: proj_weight must be TILE_LAYOUT")
    if int(sinkhorn_iters) < 1:
        raise ValueError(f"mhc_pre: sinkhorn_iters must be >= 1, got {sinkhorn_iters}")
    return n, cfg


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def mhc_pre(
    input_tensor: ttnn.Tensor,
    proj_weight: ttnn.Tensor,
    proj_bias: ttnn.Tensor,
    *,
    scale,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    norm_eps: float = 1e-6,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
):
    n, cfg = validate(
        input_tensor,
        proj_weight,
        proj_bias,
        compute_kernel_config=compute_kernel_config,
        sinkhorn_iters=sinkhorn_iters,
    )
    if len(scale) != 3:
        raise ValueError(f"mhc_pre: scale must be (a_pre, a_post, a_res), got {scale!r}")

    device = input_tensor.device()
    lead = list(input_tensor.shape)[:-1]
    C = input_tensor.shape[-1] // n

    def _alloc(last, dtype):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape(lead + [last]), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )

    y = _alloc(C, input_tensor.dtype)
    post = _alloc(n, ttnn.float32)
    comb = _alloc(n * n, ttnn.float32)

    desc, _plan = create_program_descriptor(
        input_tensor,
        proj_weight,
        proj_bias,
        y,
        post,
        comb,
        n=n,
        scale=tuple(float(s) for s in scale),
        sinkhorn_iters=int(sinkhorn_iters),
        eps=float(eps),
        norm_eps=float(norm_eps),
        cfg=cfg,
    )
    ttnn.generic_op([input_tensor, proj_weight, proj_bias, y, post, comb], desc)
    return y, post, comb
