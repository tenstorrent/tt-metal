# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_post feature spec — TARGET universe + golden-test INPUTS + INVALID.

mhc_post is the second half of one Manifold-Constrained Hyper-Connection
(mHC, DeepSeek-V4) wrap, fused into one op. Given the sublayer output
F `(..., T, C)`, the residual streams X `(..., T, n*C)` (stream i in
columns [i*C, (i+1)*C)) and the coefficients mhc_pre produced — post
`(..., T, n)` and the doubly-stochastic comb `(..., T, n*n)` — it writes
the new residual streams, per token and output stream j:

    X'_j = post_j * F + sum_i comb[i][j] * X_i

(the residual mixing applies comb transposed; still doubly stochastic).

TARGET is the planner's ambition: every axis value the op should
eventually support. SUPPORTED (in the op file) is what works now. Each
value in TARGET[axis] - SUPPORTED[axis] is a refinement candidate.

INVALID lives here (NOT in the op file) — structurally impossible cells,
skipped before the op runs.

INPUT_TAGGERS live with the op (`ttnn/operations/mhc_post/`). The op
declares one tagger, `alignment`, over the token dim T (dim -2): C is a
multiple of 32 by contract, so only T can break alignment.

Each INPUTS entry is `(F_shape, X_shape, post_shape, comb_shape)`; n is
post's last dim (4 for DeepSeek-V4). Same token/hidden shapes as mhc_pre.
"""

import ttnn

TARGET = {
    # Residual-stream dtype: X and the output X'.
    "dtype": [ttnn.float32, ttnn.bfloat16],
    # Sublayer-output (F) dtype, independent of the streams: e.g. a bf16
    # attention / MoE output added into fp32 residual streams.
    "sublayer_dtype": [ttnn.float32, ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    # DEST accumulation width: fp32 only. mHC is almost free of matmuls, so
    # 16-bit DEST buys nothing, and the coefficients must stay fp32. False
    # is outside TARGET (validate() refuses it: UnsupportedAxisValue).
    "fp32_dest_acc_en": [True],
    # Shape-derived (tagger wins; not iterated): T tile-aligned or not.
    "alignment": ["tile_aligned", "h_non_aligned"],
}


# No structural impossibilities. (post / comb are always float32 TILE —
# a fixed part of the contract, not axes.)
INVALID = []


_N = 4  # DeepSeek-V4 hc_mult


def _case(x_shape):
    lead, nc = tuple(x_shape[:-1]), x_shape[-1]
    return (lead + (nc // _N,), tuple(x_shape), lead + (_N,), lead + (_N * _N,))


# ---------------------------------------------------------------------------
# INPUTS — swept over the full TARGET cartesian (minus INVALID):
# 2 stream dtypes x 2 sublayer dtypes = 4 cells per shape; 25 shapes.
# Listed by X's shape (last dim n*C).
# ---------------------------------------------------------------------------
INPUTS = [
    # --- small, tile-aligned ---
    _case((32, 128)),  # C=32, rank 2
    _case((64, 1024)),  # C=256
    _case((1, 128, 4096)),  # C=1024, rank 3
    # --- DeepSeek-V4 prefill, per device: T = 5120-token chunk / 8-way SP ---
    _case((1, 1, 640, 7168)),  # C=1792 (hidden 7168 / TP4), rank 4
    _case((1, 1, 640, 28672)),  # C=7168 (hidden unsharded), rank 4
    # --- larger T, wide ---
    _case((2048, 16384)),  # C=4096, rank 2
    _case((1, 2, 256, 8192)),  # C=2048, batch 2, rank 4
    # --- T not a multiple of 32 ---
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
# Fixed config: TILE, sublayer_dtype = dtype, fp32_dest_acc_en=True. Swept:
# dtype {float32, bfloat16} x T x C (PERF_DIMS), where T runs over
# SP_TOKEN_COUNTS (prefill chunk 2K/4K/5K/8K split over 2/4/8 SP devices:
# 8 distinct per-device token counts).
#
# extras (perf goal, recorded — not asserted by the runner):
#   perf_regime   always "dram": n + 1 input rows in, n rows out per token,
#                 n*(n+1) multiply-adds per output element.
#   target_ns     DRAM roofline — every tensor moved ONCE:
#                   bytes = T*C*fB (read F) + T*n*C*xB (read X)
#                         + T*(n + n*n)*4 (read post, comb) + T*n*C*xB (write X')
#                   target_ns = bytes / (0.80 * 512 GB/s)
#   reference_ns  where measured: the composite ttnn implementation this op
#                 replaces (models/demos/deepseek_v3_d_p/tt/mhc/tt_mhc.py
#                 hc_post = 81 device ops), device-kernel time on Blackhole
#                 p150 (110 cores), 2026-09-29.
# ---------------------------------------------------------------------------
_DRAM_PEAK_BPS = 512e9
_DRAM_TARGET_FRAC = 0.80
_BYTES = {ttnn.float32: 4, ttnn.bfloat16: 2}

# Composite baseline (device-kernel ns), keyed (T, C, dtype).
_REFERENCE_NS = {(640, 7168, ttnn.float32): 3_780_700}

# PERF FOCUS — mandatory targets for the trailing perf passes. All bf16
# streams, the production precision (mHC paper §4.3.1; DeepSeek-V4 and
# GLM-5.3 run bf16 residual streams): DeepSeek-V4 per device unsharded and
# under TP4, and GLM-5.3-Flash per chip. float32 cases are measured but not
# a focus (they are the accuracy ceiling, and carry the composite baseline).
_PERF_FOCUS = {(640, 7168, ttnn.bfloat16), (640, 1792, ttnn.bfloat16), (1280, 4096, ttnn.bfloat16)}


def perf_target(T, C, *, dtype, sublayer_dtype=None, n=_N):
    """DRAM-roofline goal for one mhc_post call — see the block above."""
    xb = _BYTES[dtype]
    fb = _BYTES[sublayer_dtype or dtype]
    nbytes = T * C * fb + 2 * T * n * C * xb + T * (n + n * n) * 4
    return {
        "perf_regime": "dram",
        "target_ns": int(round(nbytes / (_DRAM_PEAK_BPS * _DRAM_TARGET_FRAC) * 1e9)),
        "expected_dram_util": _DRAM_TARGET_FRAC,
    }


def _perf_case(T, C, dtype):
    chunks = ", ".join(f"{c // 1024}K/SP{sp}" for c, sp in SP_TOKEN_COUNTS[T])
    tag = f" {_MODEL_TAG[C]}" if C in _MODEL_TAG else ""
    extras = {"label": f"mhc_post T{T} C{C}{tag} ({chunks})", **perf_target(T, C, dtype=dtype)}
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
        "sublayer_dtype": dtype,
        "layout": ttnn.TILE_LAYOUT,
        "fp32_dest_acc_en": True,
        "extras": extras,
    }


LOOSE_CASES = [
    _perf_case(T, C, dtype) for dtype in (ttnn.float32, ttnn.bfloat16) for T in SP_TOKEN_COUNTS for C in PERF_DIMS
]
