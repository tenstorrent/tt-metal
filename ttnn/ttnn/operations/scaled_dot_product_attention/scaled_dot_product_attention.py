# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""scaled_dot_product_attention: prefill attention softmax(scale * Q @ K^T + mask) @ V on Quasar.

One `ttnn.generic_op` dispatch of one `ttnn.ProgramSpec` (FlashAttention form: a block of query
tile-rows walks the keys in chunks with a running max / sum / output). The program itself is built in
`scaled_dot_product_attention_program_artifacts.py`; this file holds the registry declarations
(INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate()) and the public entry point.

Specification: eval/specs/scaled_dot_product_attention/spec.py (RULES R1-R14, REJECTIONS X1-X6).
"""

from __future__ import annotations

import math
import struct

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval import memory
from eval.specs.scaled_dot_product_attention.spec import AXES  # noqa: E402

from .scaled_dot_product_attention_program_artifacts import (
    THREAD_PLAN,
    create_program_artifacts,
    make_plan,
)

OP_NAME = "scaled_dot_product_attention"

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase, equal to the specification's PHASE0 axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout": [ttnn.TILE_LAYOUT],
    "memory_config": ["DRAM"],
    "sharding": ["interleaved"],
    "is_causal": [False, True],
    "kv_heads_mode": ["equal", "grouped"],
    "batch": ["single"],
    "alignment": ["tile_aligned"],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: none in the first phase.
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# The compute configuration (rules R5, R6, R7)
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means, in one exported place (rule R5)."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _resolve_compute_config(compute_kernel_config):
    """A ttnn.ComputeConfigDescriptor built from the three fields of any config object (rule R7): a
    ComputeConfigDescriptor or tt-metal's device config (WormholeComputeKernelConfig) alike. Fields a
    descriptor lacks (packer_l1_acc, dst_full_sync_en, throttle_level) are ignored."""
    config = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=config.math_fidelity,
        fp32_dest_acc_en=bool(config.fp32_dest_acc_en),
        math_approx_mode=bool(config.math_approx_mode),
    )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _shape(t):
    return tuple(int(d) for d in t.shape)


def _check_rejections(input_tensor_q, input_tensor_k, input_tensor_v):
    """REJECTIONS X1-X6: malformed calls (ValueError)."""
    shapes = {"input_tensor_q": _shape(input_tensor_q), "input_tensor_k": _shape(input_tensor_k),
              "input_tensor_v": _shape(input_tensor_v)}
    for name, shape in shapes.items():
        if len(shape) != 4:  # X1
            raise ValueError(f"{OP_NAME}: {name} must have four dimensions [B, H, S, D], got {list(shape)}")
    q, k, v = shapes["input_tensor_q"], shapes["input_tensor_k"], shapes["input_tensor_v"]
    if k != v:  # X3
        raise ValueError(f"{OP_NAME}: the keys {list(k)} and the values {list(v)} must have the same shape")
    if q[0] != k[0]:  # X6
        raise ValueError(f"{OP_NAME}: the keys hold {k[0]} sequences, the queries {q[0]}")
    if q[2] != k[2]:  # X4
        raise ValueError(f"{OP_NAME}: the keys have {k[2]} tokens, the queries {q[2]} (cross-attention is not this op)")
    if q[3] != k[3]:  # X5
        raise ValueError(f"{OP_NAME}: the keys have head dimension {k[3]}, the queries {q[3]}")
    if k[1] == 0 or q[1] % k[1] != 0:  # X2
        raise ValueError(f"{OP_NAME}: {q[1]} query heads is not a multiple of {k[1]} key/value heads")


def _check_kv_placement(input_tensor_q, input_tensor_k, input_tensor_v):
    """Rule R10: keys and values interleaved in DRAM, tile layout, the queries' dtype."""
    for name, t in (("input_tensor_k", input_tensor_k), ("input_tensor_v", input_tensor_v)):
        placement = memory.memory_axes(t)
        if placement != {"memory_config": "DRAM", "sharding": "interleaved"}:
            raise NotImplementedError(f"{OP_NAME}: {name} must be interleaved in DRAM, it is {placement}")
        if t.layout != ttnn.TILE_LAYOUT:
            raise NotImplementedError(f"{OP_NAME}: {name} must be in TILE_LAYOUT, it is {t.layout}")
        if t.dtype != input_tensor_q.dtype:
            raise NotImplementedError(
                f"{OP_NAME}: {name} must have the queries' dtype {input_tensor_q.dtype}, it is {t.dtype}")


def _axes_of(input_tensor_q, input_tensor_k, input_tensor_v, is_causal, scale, config):
    axes = {
        "dtype": input_tensor_q.dtype,
        "layout": input_tensor_q.layout,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "is_causal": bool(is_causal),
        **memory.memory_axes(input_tensor_q),  # rule R9
    }
    inputs = (_shape(input_tensor_q), _shape(input_tensor_k), _shape(input_tensor_v), scale)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)
    return axes


def validate(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    *,
    is_causal=True,
    scale=None,
    program_config=None,
    compute_kernel_config=None,
):
    """The runtime gate. Reads neither math_fidelity nor math_approx_mode (rule R6) nor program_config
    (rule R8). Raises ValueError for malformed calls (X1-X6), NotImplementedError for anything outside the
    current phase (R4, R9, R10, R14)."""
    _check_rejections(input_tensor_q, input_tensor_k, input_tensor_v)
    _check_kv_placement(input_tensor_q, input_tensor_k, input_tensor_v)
    config = _resolve_compute_config(compute_kernel_config)
    axes = _axes_of(input_tensor_q, input_tensor_k, input_tensor_v, is_causal, scale, config)

    # Rule R4: float32 with a 16-bit accumulator is legal but lossy; refused in every phase.
    if axes["dtype"] == ttnn.float32 and not axes["fp32_dest_acc_en"]:
        raise UnsupportedAxisValue(f"{OP_NAME}: float32 inputs need fp32_dest_acc_en=True (rule R4)")

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{OP_NAME}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"{OP_NAME}: unsupported combination (refinement candidate): {exc}")
    return config


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _scale_bits(scale, head_dim):
    """`scale` as the fp32 bit pattern the compute kernel multiplies by; None is 1 / sqrt(D) (rule R3)."""
    value = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)
    return struct.unpack("I", struct.pack("f", value))[0]


def scaled_dot_product_attention(
    input_tensor_q: ttnn.Tensor,
    input_tensor_k: ttnn.Tensor,
    input_tensor_v: ttnn.Tensor,
    *,
    is_causal: bool = True,
    scale: float = None,
    program_config=None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    """softmax(scale * Q @ K^T + mask) @ V, the mask causal when `is_causal`. `program_config` is a hint the
    op ignores (rule R8)."""
    return run_attention(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        is_causal=is_causal,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def run_attention(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    *,
    is_causal=True,
    scale=None,
    program_config=None,
    compute_kernel_config=None,
    threads=THREAD_PLAN,
    q_chunk_tiles=None,
    k_chunk_tiles=None,
    tag_threads=False,
):
    """The entry point with the test probes: the thread plan (R, C, W), knob overrides of the blocking scheme
    (clamped to legality, never above the fit), and `tag_threads`, which makes every NEO write its id + 1 instead
    of the result (test_each_neo_computes_its_own_unit). validate() is the first statement."""
    config = validate(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        is_causal=is_causal,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    device = input_tensor_q.device()
    plan = make_plan(
        input_tensor_q,
        input_tensor_k,
        is_causal=bool(is_causal),
        fp32_dest_acc_en=bool(config.fp32_dest_acc_en),
        threads=threads,
        q_chunk_tiles=q_chunk_tiles,
        k_chunk_tiles=k_chunk_tiles,
    )
    # Rule R11: the output is interleaved in DRAM whatever buffer the queries are in.
    out_spec = ttnn.TensorSpec(
        list(input_tensor_q.shape), input_tensor_q.dtype, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    output = ttnn.allocate_tensor_on_device(out_spec, device)
    spec, run_args, tensor_indices = create_program_artifacts(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        output,
        plan=plan,
        scale_bits=_scale_bits(scale, int(input_tensor_q.shape[-1])),
        compute_config=config,
        tag_threads=tag_threads,
    )
    return ttnn.generic_op([input_tensor_q, input_tensor_k, input_tensor_v, output], spec, run_args, tensor_indices)
