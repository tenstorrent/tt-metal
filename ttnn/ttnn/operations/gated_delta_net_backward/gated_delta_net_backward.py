# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""gated_delta_net_backward — vector-Jacobian product of the chunked gated delta rule.

One native TTNN device-program dispatch (`ttnn.generic_op`) computes all six
gradients.  The program is internally phased into three stages separated by two
per-group semaphore rendezvous (see `op_design.md` "Block schedule"):

    P : prep       — grid-parallel over (bh, chunk); builds the decay vectors,
                     the intra-chunk decay mask L, the UT-transform inverse
                     Tinv, and the state-independent scan seeds u and c.
    S : scan       — one core per (b,h); the forward state scan followed by the
                     reverse state-gradient scan, sequential in the chunk index.
    G : assembly   — grid-parallel over (bh, chunk); the six gradients.

Registry-model op file: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate().
INVALID is deliberately absent — it lives in the golden suite's feature_spec.

KNOWN PRECISION FLOOR (measured, one golden cell).  `dg` is the op's weakest
gradient by construction -- it is the only one through `exp()` of a cumulative
sum -- and at the saturated-gate setting (`g_scale = 8`, where the state is
wiped every step and `dg` is numerically ~1e-3 while the other gradients are
O(1)) it lands at PCC 0.99974 / rel-RMS 0.024 against the float64 oracle,
inside the PCC band but outside the 0.02 RMS band.

The term was isolated with a scratch-readback harness: recomputing `dg` on the
host in float64 from the DEVICE's own scratch reproduces the miss (0.026), and
substituting only the exact `decay` column removes it (0.002).  `decay` is a
cumulative sum reaching |242| at that setting, and every consumer of it is
`exp(decay[t] - decay[s])`.

WHERE THE BITS GO (corrected during verification; the earlier reading here
blamed the packer and it is not the packer).  A `Float32` CB page is a full
float32 store -- but `build_L()` forms the difference matrix as `X - X^T` with
`X = decay (x) 1` produced by an outer-product MATMUL, so `decay` enters
through an FPU SOURCE register at ~tf32 width (10 explicit mantissa bits,
i.e. ~0.2 absolute resolution at |242|).  The loss is at the consumer, which is
why keeping the cumsum in fp32 in L1 -- or computing it in the SFPU -- changes
nothing, and why the fix is to stop making a large-magnitude value an FPU
operand at all: `D = LT @ diag(g) @ strict_lower` is algebraically identical,
has `g` (small) as its only data operand, and models 5.8x better on L's
relative RMS at this setting.  Filed as Refinement 1 in `op_requirements.md`
with the measurement; it is a refinement, not a bug fix.

CALLER CONTRACT: `q` and `k` MUST arrive L2-normalized along their last
dimension.  With un-normalized q/k the UT transform's forward substitution is
not contractive and the *forward* diverges (measured |o|max of 2.9e18); this op
differentiates that forward and inherits the requirement.  It is not something
the op fixes.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .gated_delta_net_backward_program_descriptor import build_program


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# `inputs` is `((B, T, H, K, V), chunk_size)` — chunk_size travels WITH the
# shape because the interesting distinction (`T % chunk_size == 0`) is a
# property of the pair.  Declaration order matters: `chunk_size` first so
# `tag_seq_alignment` can read `axes["chunk_size"]`.


def tag_chunk_size(inputs, axes):
    """The per-shape chunk size."""
    return int(inputs[1])


def tag_seq_alignment(inputs, axes):
    """Whether the sequence divides evenly into chunks."""
    t = int(inputs[0][1])
    c = int(axes["chunk_size"])
    return "chunk_aligned" if t % c == 0 else "chunk_ragged"


def tag_head_dims(inputs, axes):
    """`square` when K == V, `wide_v` when V > K (the GVA / Qwen3.5 geometry)."""
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
#
# `bfloat8_b` is absent on purpose and is NOT a refinement candidate: it is
# INVALID at this op's TARGET (block-quantized gate sequences — see the golden
# suite's feature_spec.py).

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "state_mode": ["do_only", "with_h0", "with_h0_and_dht"],
    "chunk_size": [32, 64],
    "seq_alignment": ["chunk_aligned", "chunk_ragged"],
    "head_dims": ["square", "wide_v"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
    "math_fidelity": {"value": ["LoFi", "HiFi2", "HiFi3", "HiFi4"], "source": "declared"},
}


# ---------------------------------------------------------------------------
# Mechanism caps that are NOT support axes
# ---------------------------------------------------------------------------
#
# `H <= 32`: the source page-index formula assumes ceil(H / 32) == 1, so one
# head is one row of every page.  Every INPUTS entry has H <= 8.
MAX_HEADS = 32


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(q, k, v, g, beta, do, *, dht=None, initial_state=None, chunk_size=64, **_):
    qs = [int(d) for d in q.shape]
    vs = [int(d) for d in v.shape]
    if len(qs) != 4 or len(vs) != 4:
        raise ValueError("gated_delta_net_backward: q/k must be [B,T,H,K] and v/do [B,T,H,V]")
    B, T, H, K = qs[0], qs[1], qs[2], qs[3]
    V = vs[3]

    if initial_state is None:
        state_mode = "do_only"
    elif dht is None:
        state_mode = "with_h0"
    else:
        state_mode = "with_h0_and_dht"

    axes = {
        "dtype": q.dtype,
        "layout": q.layout,
        "state_mode": state_mode,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(((B, T, H, K, V), chunk_size), axes)

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"gated_delta_net_backward: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside SUPPORTED
    for exc in EXCLUSIONS:
        if all(axes.get(key) == val for key, val in exc.items()):
            raise ExcludedCell(f"gated_delta_net_backward: unsupported combination (refinement candidate): {exc}")

    # 3. Mechanism caps (not support axes — hard errors, not refusals)
    if int(chunk_size) % 32 != 0:
        raise ValueError(f"gated_delta_net_backward: chunk_size must be a multiple of 32, got {chunk_size}")
    if H > MAX_HEADS:
        raise ValueError(
            f"gated_delta_net_backward: H={H} exceeds the mechanism cap of {MAX_HEADS} "
            "(the source page-index formula assumes ceil(H/32) == 1)"
        )

    # Every tensor must agree on dtype / layout — the taggers only saw q.
    others = [("k", k), ("v", v), ("g", g), ("beta", beta), ("do", do)]
    if initial_state is not None:
        others.append(("initial_state", initial_state))
    if dht is not None:
        others.append(("dht", dht))
    for name, tensor in others:
        if tensor.dtype != q.dtype:
            raise UnsupportedAxisValue(
                f"gated_delta_net_backward: mixed dtypes — {name}.dtype={tensor.dtype} != q.dtype={q.dtype}"
            )
        if tensor.layout != q.layout:
            raise UnsupportedAxisValue(
                f"gated_delta_net_backward: mixed layouts — {name}.layout={tensor.layout} != q.layout={q.layout}"
            )

    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Debug hook: the last dispatch's internal scratch tensors, keyed by name.
# Populated on every call so a debug test can read the device's intermediate
# blocks back with `ttnn.to_torch` and compare them against the reference.
# Not part of the public contract.
_LAST_SCRATCH = {}


def gated_delta_net_backward(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    g: ttnn.Tensor,
    beta: ttnn.Tensor,
    do: ttnn.Tensor,
    *,
    dht: ttnn.Tensor = None,
    initial_state: ttnn.Tensor = None,
    chunk_size: int = 64,
    scale: float = None,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = None,
):
    """Backward pass (VJP) of the chunked gated delta rule.

    Args:
        q, k: `[B, T, H, K]`, L2-normalized along K (caller contract).
        v: `[B, T, H, V]`.
        g: `[B, T, H]` log-space decay gate (<= 0).
        beta: `[B, T, H]` delta-rule write strength, in (0, 1).
        do: `[B, T, H, V]` gradient of the output.
        dht: `[B, H, K, V]` gradient of the final state, or None.
        initial_state: `[B, H, K, V]`, or None.
        chunk_size: tokens per chunk; a multiple of 32.
        scale: query scale; defaults to `K ** -0.5`.
        compute_kernel_config: precision knobs (math fidelity, fp32 dest accum).
        memory_config: placement of the outputs; defaults to q's.

    Returns:
        `(dq, dk, dv, dg, dbeta, dh0)`.  `dh0` is `None` — not a zero tensor —
        when `initial_state is None`.
    """
    validate(
        q,
        k,
        v,
        g,
        beta,
        do,
        dht=dht,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )

    global _LAST_SCRATCH
    outputs, scratch = build_program(
        q,
        k,
        v,
        g,
        beta,
        do,
        dht=dht,
        initial_state=initial_state,
        chunk_size=int(chunk_size),
        scale=scale,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )
    _LAST_SCRATCH = scratch
    return outputs
