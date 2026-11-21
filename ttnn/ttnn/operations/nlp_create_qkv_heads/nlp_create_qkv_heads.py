# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""nlp_create_qkv_heads: cut a fused QKV row tensor [B, 1, S, (nh + 2*nkv)*hd] into the heads of q, k and v.

Registry model (eval/REGISTRY_MODEL.md): INPUT_TAGGERS (the specification's AXES), SUPPORTED, EXCLUSIONS,
validate(), and the public entry point, whose signature equals the specification's stub
(eval/specs/nlp_create_qkv_heads/spec.py). One `ttnn.generic_op` dispatch of one `ttnn.ProgramSpec` per call.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval import memory
from eval.specs.nlp_create_qkv_heads.spec import AXES, canonical_memory, head_shapes  # noqa: E402

from .nlp_create_qkv_heads_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase (= the specification's PHASE0).
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":             [ttnn.bfloat16],
    "layout":            [ttnn.TILE_LAYOUT],
    "memory_config":     ["DRAM"],
    "sharding":          ["interleaved"],
    "transpose_k_heads": [False],
    "output_memory":     ["DRAM"],
    "alignment":         ["tile_aligned"],
    "batch":             ["single"],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _kv_heads(num_heads, num_kv_heads):
    return num_heads if num_kv_heads is None else num_kv_heads


def _is_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _check_structure(input, num_heads, num_kv_heads):
    """The malformed calls (REJECTIONS X1 to X4), before any axis is derived."""
    shape = tuple(input.shape)
    if len(shape) != 4 or shape[1] != 1:
        raise ValueError(f"nlp_create_qkv_heads: input must be [B, 1, S, width], got {list(shape)}")
    if not _is_count(num_heads):
        raise ValueError(f"nlp_create_qkv_heads: num_heads must be an integer >= 1, got {num_heads!r}")
    if num_kv_heads is not None and not _is_count(num_kv_heads):
        raise ValueError(f"nlp_create_qkv_heads: num_kv_heads must be None or an integer >= 1, got {num_kv_heads!r}")
    sections = num_heads + 2 * _kv_heads(num_heads, num_kv_heads)
    if shape[3] % sections != 0 or shape[3] < sections:
        raise ValueError(
            f"nlp_create_qkv_heads: width {shape[3]} is not a multiple of num_heads + 2 * num_kv_heads = {sections}"
        )


def axes_of(input, *, num_heads, num_kv_heads=None, transpose_k_heads=True, memory_config=None):
    """The cell of a call: tensor properties, placement (R8), arguments and every tagger."""
    axes = {
        "dtype": input.dtype,
        "layout": input.layout,
        **memory.memory_axes(input),  # "memory_config" and "sharding" of the input (R8)
        "transpose_k_heads": bool(transpose_k_heads),
        "output_memory": canonical_memory(memory_config, None),  # TypeError on a non-config (X5), R6
    }
    inputs = (tuple(input.shape), num_heads, num_kv_heads)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)
    return axes


def validate(input, *, num_heads, num_kv_heads=None, transpose_k_heads=True, memory_config=None):
    _check_structure(input, num_heads, num_kv_heads)
    axes = axes_of(
        input,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=transpose_k_heads,
        memory_config=memory_config,
    )
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"nlp_create_qkv_heads: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"nlp_create_qkv_heads: unsupported combination (refinement candidate): {exc}")
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_BUFFER_TYPES = {"DRAM": ttnn.BufferType.DRAM, "L1": ttnn.BufferType.L1}


def output_shapes(input, num_heads, num_kv_heads, transpose_k_heads):
    """(q, k, v) shapes: the specification's `head_shapes`, the single definition of the result shapes."""
    inputs = (tuple(input.shape), num_heads, num_kv_heads)
    return tuple(head_shapes(inputs, {"transpose_k_heads": bool(transpose_k_heads)}))


def nlp_create_qkv_heads(
    input: ttnn.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int = None,
    transpose_k_heads: bool = True,
    memory_config: "ttnn.MemoryConfig" = None,
) -> list:
    """[q, k, v]: q [B, num_heads, S, head_dim], k and v [B, num_kv_heads, S, head_dim], cut from the input's
    rows (eval/specs/nlp_create_qkv_heads/spec.py, DEFINITION). One device-program dispatch."""
    axes = validate(
        input,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=transpose_k_heads,
        memory_config=memory_config,
    )
    device = input.device()
    # R5: each result its own interleaved tensor, in the buffer memory_config names, else the input's.
    buffer = axes["output_memory"] if axes["output_memory"] is not None else axes["memory_config"]
    outputs = [
        ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(ttnn.Shape(shape), input.dtype, ttnn.TILE_LAYOUT, buffer_type=_BUFFER_TYPES[buffer]),
            device,
        )
        for shape in output_shapes(input, num_heads, num_kv_heads, transpose_k_heads)
    ]
    q, k, v = outputs
    spec, run_args, tensor_indices = create_program_artifacts(
        input, q, k, v, num_heads=num_heads, num_kv_heads=_kv_heads(num_heads, num_kv_heads)
    )
    # generic_op returns only the last io tensor; the op returns its own list (R2).
    ttnn.generic_op([input, q, k, v], spec, run_args, tensor_indices)
    return [q, k, v]
