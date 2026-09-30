# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_pre feature spec — TARGET universe + golden-test INPUTS + INVALID.

mhc_pre is the first half of one Manifold-Constrained Hyper-Connection
(mHC, DeepSeek-V4) wrap, fused into one op. From the n packed residual
streams X `(..., T, n*C)` it computes, per token:

    mixes = RMSNorm(X) @ W                     (W: (n*C, n*(n+2)), no norm weight)
    pre   = sigmoid(a_pre  * mixes[0:n]    + b[0:n])    + eps
    post  = 2 * sigmoid(a_post * mixes[n:2n] + b[n:2n])
    comb  = Sinkhorn(a_res * mixes[2n:] + b[2n:])       (n x n, doubly stochastic)
    y     = sum_i pre_i * X_i                  (X_i = stream i, columns [i*C, (i+1)*C))

and returns (y, post, comb). The sublayer F (attention / MoE) then runs on
y, and mhc_post consumes (F(y), X, post, comb).

TARGET is the planner's ambition: every axis value the op should
eventually support. SUPPORTED (in the op file) is what works now. Each
value in TARGET[axis] - SUPPORTED[axis] is a refinement candidate.

INVALID lives here (NOT in the op file) — structurally impossible cells,
skipped before the op runs.

INPUT_TAGGERS live with the op (`ttnn/operations/mhc_pre/`). The op
declares one tagger, `alignment`, over the token dim T (dim -2): C (and
so n*C) is a multiple of 32 by contract, so only T can break alignment.

Each INPUTS entry is `(X_shape, W_shape)`. X is rank 2/3/4 — every dim but
the last folds into the token count T. W is always 2D `(n*C, n*(n+2))`;
n is derived from W's last dim (24 -> n=4). Shapes are DeepSeek-V4-derived
(C = 7168 full, 1792 = per device under 4-way TP of the hidden dim) plus
small / odd-T coverage shapes.
"""

import ttnn

TARGET = {
    # Residual-stream dtype: X and y. The coefficient outputs (post, comb)
    # are ALWAYS float32, whatever the stream dtype.
    "dtype": [ttnn.float32, ttnn.bfloat16],
    # X and y are TILE only: the streams are packed along the last dim and
    # every stream boundary is a tile boundary (C % 32 == 0).
    "layout": [ttnn.TILE_LAYOUT],
    # Projection weight W dtype (always TILE). bf16 halves the weight read.
    "weight_dtype": [ttnn.float32, ttnn.bfloat16],
    # DEST accumulation width: fp32 only. mHC is almost free of matmuls, so
    # 16-bit DEST buys nothing, and the coefficients must stay fp32. False
    # is outside TARGET (validate() refuses it: UnsupportedAxisValue).
    "fp32_dest_acc_en": [True],
    # Shape-derived (tagger wins; not iterated): T tile-aligned or not.
    "alignment": ["tile_aligned", "h_non_aligned"],
}


# No structural impossibilities: both dtypes are TILE-representable, and
# the weight and stream axes describe independent tensors.
INVALID = []


_N = 4  # DeepSeek-V4 hc_mult
_MIX = _N * (_N + 2)  # 24 = pre (n) + post (n) + comb (n*n)


def _case(x_shape):
    return (tuple(x_shape), (x_shape[-1], _MIX))


# ---------------------------------------------------------------------------
# INPUTS — swept over the full TARGET cartesian (minus INVALID):
# 2 stream dtypes x 2 weight dtypes = 4 cells per shape; 25 shapes.
# Last dim is n*C (C = last // 4).
# ---------------------------------------------------------------------------
INPUTS = [
    # --- small, tile-aligned ---
    _case((32, 128)),  # C=32, one tile of tokens, rank 2
    _case((64, 1024)),  # C=256
    _case((1, 128, 4096)),  # C=1024, rank 3
    # --- DeepSeek-V4 prefill, per device: T = 5120-token chunk / 8-way SP ---
    _case((1, 1, 640, 7168)),  # C=1792 (hidden 7168 / TP4), rank 4
    _case((1, 1, 640, 28672)),  # C=7168 (hidden unsharded), rank 4
    # --- larger T, wide ---
    _case((2048, 16384)),  # C=4096, rank 2
    _case((1, 2, 256, 8192)),  # C=2048, batch 2, rank 4
    # --- T not a multiple of 32 (last prefill chunk, decode) ---
    _case((17, 512)),  # C=128
    _case((1, 100, 4096)),  # C=1024, rank 3
    _case((1, 1, 1000, 7168)),  # C=1792
    _case((1, 28672)),  # T=1 decode token, C=7168
    # --- DeepSeek-V4 prefill chunk / SP split, per device, C=1792 (TP4) ---
    # T = chunk / SP for chunk in {2K, 4K, 5K, 8K}, SP in {2, 4, 8}
    # (640 = 5K / 8 is listed above).
    *[_case((1, 1, T, 4 * 1792)) for T in (256, 512, 1024, 1280, 2048, 2560, 4096)],
    # --- common embedding dims C (EMB_DIMS), unsharded ---
    _case((1, 1, 1280, 4 * 2560)),  # C=2560
    _case((1, 1, 1280, 4 * 4096)),  # C=4096: GLM-5.3-Flash per chip (5K chunk / 4, bf16 streams)
    _case((1, 1, 512, 4 * 5120)),  # C=5120
    _case((1, 1, 256, 4 * 6144)),  # C=6144
    _case((1, 1, 2048, 4 * 7168)),  # C=7168
    _case((1, 77, 4 * 2560)),  # C=2560, T not a multiple of 32, rank 3
    _case((1, 1, 333, 4 * 6144)),  # C=6144, T not a multiple of 32
]

# Common embedding dims the op must serve at full speed, plus the DeepSeek-V4
# hidden under 4-way TP (7168 / 4). Known mHC users: 4096 GLM-5.3-Flash,
# 7168 DeepSeek-V4.
EMB_DIMS = (2560, 4096, 5120, 6144, 7168)
PERF_DIMS = EMB_DIMS + (1792,)
_MODEL_TAG = {7168: "DeepSeek-V4", 1792: "DeepSeek-V4 TP4", 4096: "GLM-5.3-Flash"}

# Per-device token counts of the common prefill chunk sizes under 2/4/8-way
# sequence parallelism: T -> every (chunk, SP) pair that yields it.
PREFILL_CHUNKS = (2048, 4096, 5120, 8192)
SP_SPLITS = (2, 4, 8)
SP_TOKEN_COUNTS = {}
for _chunk in PREFILL_CHUNKS:
    for _sp in SP_SPLITS:
        SP_TOKEN_COUNTS.setdefault(_chunk // _sp, []).append((_chunk, _sp))
SP_TOKEN_COUNTS = dict(sorted(SP_TOKEN_COUNTS.items()))


# ---------------------------------------------------------------------------
# LOOSE_CASES — perf sweep at the PERF-TARGET configuration.
#
# Fixed config: TILE X, float32 W, fp32_dest_acc_en=True (the Sinkhorn is
# fp32-sensitive). Swept: stream dtype {float32, bfloat16} x T x C (PERF_DIMS),
# where T runs over SP_TOKEN_COUNTS (prefill chunk 2K/4K/5K/8K split
# over 2/4/8 SP devices: 8 distinct per-device token counts).
#
# extras (perf goal, recorded — not asserted by the runner):
#   perf_regime   always "dram": every byte of X is read once and the math
#                 per byte is small (a 24-wide projection, a 4-term mix).
#   target_ns     DRAM roofline — every tensor moved ONCE:
#                   bytes = T*n*C*xB (read X) + n*C*24*wB (read W)
#                         + T*C*xB (write y) + T*(n + n*n)*4 (write post, comb)
#                   target_ns = bytes / (0.80 * 512 GB/s)
#                 Moving X once is achievable: the per-device X of the V4
#                 shapes (<= 73 MB fp32 at T=640) fits in aggregate L1 when
#                 split over the grid, so the projection (which needs the
#                 whole row before `pre` exists) and the y-mix need not
#                 re-read DRAM. Where X exceeds aggregate L1 (e.g. T=5120,
#                 C=7168, fp32 = 587 MB) a design that reads X twice is
#                 expected, and target_ns is a floor rather than a goal.
#   reference_ns  where measured: the composite ttnn implementation this op
#                 replaces (models/demos/deepseek_v3_d_p/tt/mhc/tt_mhc.py,
#                 project + mhc_split_sinkhorn + hc_pre = 25 device ops),
#                 device-kernel time on Blackhole p150 (110 cores), 2026-09-29.
# ---------------------------------------------------------------------------
_DRAM_PEAK_BPS = 512e9
_DRAM_TARGET_FRAC = 0.80
_BYTES = {ttnn.float32: 4, ttnn.bfloat16: 2}

# Composite baseline (device-kernel ns), keyed (T, C, dtype).
_REFERENCE_NS = {(640, 7168, ttnn.float32): 2_395_600}

# PERF FOCUS — mandatory targets for the trailing perf passes. All bf16
# streams, the production precision (mHC paper §4.3.1; DeepSeek-V4 and
# GLM-5.3 run bf16 residual streams): DeepSeek-V4 per device unsharded and
# under TP4, and GLM-5.3-Flash per chip. float32 cases are measured but not
# a focus (they are the accuracy ceiling, and carry the composite baseline).
_PERF_FOCUS = {(640, 7168, ttnn.bfloat16), (640, 1792, ttnn.bfloat16), (1280, 4096, ttnn.bfloat16)}


def perf_target(T, C, *, dtype, weight_dtype=ttnn.float32, n=_N):
    """DRAM-roofline goal for one mhc_pre call — see the block above."""
    xb, wb = _BYTES[dtype], _BYTES[weight_dtype]
    nbytes = T * n * C * xb + n * C * n * (n + 2) * wb + T * C * xb + T * (n + n * n) * 4
    return {
        "perf_regime": "dram",
        "target_ns": int(round(nbytes / (_DRAM_PEAK_BPS * _DRAM_TARGET_FRAC) * 1e9)),
        "expected_dram_util": _DRAM_TARGET_FRAC,
    }


def _perf_case(T, C, dtype):
    chunks = ", ".join(f"{c // 1024}K/SP{sp}" for c, sp in SP_TOKEN_COUNTS[T])
    tag = f" {_MODEL_TAG[C]}" if C in _MODEL_TAG else ""
    extras = {"label": f"mhc_pre T{T} C{C}{tag} ({chunks})", **perf_target(T, C, dtype=dtype)}
    if (T, C, dtype) in _REFERENCE_NS:
        extras["reference_ns"] = _REFERENCE_NS[(T, C, dtype)]
    if (T, C, dtype) in _PERF_FOCUS:
        extras["attention"] = (
            "PERF FOCUS — mandatory target shape for the trailing perf passes; "
            "optimize this exact config toward extras.target_ns"
        )
    return {
        "inputs": _case((1, 1, T, _N * C)),
        "dtype": dtype,
        "layout": ttnn.TILE_LAYOUT,
        "weight_dtype": ttnn.float32,
        "fp32_dest_acc_en": True,
        "extras": extras,
    }


LOOSE_CASES = [
    _perf_case(T, C, dtype) for dtype in (ttnn.float32, ttnn.bfloat16) for T in SP_TOKEN_COUNTS for C in PERF_DIMS
]
