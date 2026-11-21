# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rotary_embedding_llama: decode-mode Llama rotary embedding, `out = x * cos + (x @ T) * sin`.

Specification: eval/specs/rotary_embedding_llama/spec.py. Design: op_design.md (this directory).

All four inputs are height-sharded in L1 on the same cores and the result is too, with the input's own
memory config (R9). The op works on the shards where they lie: one ttnn.generic_op dispatch of a Metal 2.0
ProgramSpec whose rings are borrowed from the shards (rotary_embedding_llama_program_artifacts.py).
"""

from __future__ import annotations

import ttnn

from eval import memory
from eval.specs.rotary_embedding_llama.spec import AXES
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rotary_embedding_llama_program_artifacts import TILE, create_program_artifacts

OP_NAME = "rotary_embedding_llama"

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's AXES, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase is the specification's PHASE0.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":            [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout":           [ttnn.TILE_LAYOUT],
    "memory_config":    ["L1"],
    "sharding":         ["height"],
    "is_decode_mode":   [True],
    "users":            [1],
    "alignment":        ["tile_aligned", "h_non_aligned"],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # R1: float32 with a 16-bit accumulator is legal but lossy, and the op refuses it natively
    # (eval/CONVENTIONS.md: refused op-side through EXCLUSIONS, not INVALID). Inert while float32 is outside
    # SUPPORTED["dtype"]; it takes effect the moment float32 is added.
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]

# The shape limits of the specification's REJECTIONS (X4, X6, X7): a head row is one run of 32-channel
# blocks, at most 128 wide, and a user's heads are one tile row.
MAX_HEAD_DIM = 128
MAX_HEADS = TILE


# ---------------------------------------------------------------------------
# Compute config (R5, R6, R7)
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means: the one place the default lives (R5)."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _resolve_compute_config(compute_kernel_config):
    """A ttnn.ComputeConfigDescriptor from whatever the caller passed (R7): a descriptor, or tt-metal's device
    compute kernel config (ttnn.WormholeComputeKernelConfig). Its three fields are read as attributes; fields a
    descriptor does not have (packer_l1_acc, throttle_level) are ignored. No field is gated on here (R6)."""
    if compute_kernel_config is None:
        return default_compute_kernel_config()
    default = default_compute_kernel_config()
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(compute_kernel_config, "math_fidelity", default.math_fidelity),
        fp32_dest_acc_en=bool(getattr(compute_kernel_config, "fp32_dest_acc_en", default.fp32_dest_acc_en)),
        math_approx_mode=bool(getattr(compute_kernel_config, "math_approx_mode", default.math_approx_mode)),
    )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _shape(tensor):
    return tuple(int(d) for d in tensor.shape)


def _check_shapes(x, cos, sin, trans_mat):
    """The malformed calls of the specification's REJECTIONS, refused on shape before anything else."""
    xs, cs, ss, ts = _shape(x), _shape(cos), _shape(sin), _shape(trans_mat)
    if len(xs) != 4:
        raise ValueError(f"{OP_NAME}: input_tensor must be rank 4 [1, users, heads, head_dim], got {list(xs)}")
    batch, users, heads, head_dim = xs
    if batch != 1:
        raise ValueError(f"{OP_NAME}: decode mode has one token a user: input_tensor dim 0 must be 1, got {batch}")
    if head_dim % TILE != 0:
        raise ValueError(f"{OP_NAME}: head_dim ({head_dim}) must be a multiple of {TILE}: T acts on runs of 32 channels")
    if head_dim > MAX_HEAD_DIM:
        raise ValueError(f"{OP_NAME}: head_dim ({head_dim}) must be at most {MAX_HEAD_DIM}")
    if heads > MAX_HEADS:
        raise ValueError(f"{OP_NAME}: at most {MAX_HEADS} head rows a user (one tile row), got {heads}")
    if cs != ss:
        raise ValueError(f"{OP_NAME}: cos_cache {list(cs)} and sin_cache {list(ss)} differ in shape")
    if cs != (1, users, 1, head_dim):
        raise ValueError(f"{OP_NAME}: cos_cache and sin_cache must be [1, {users}, 1, {head_dim}], got {list(cs)}")
    if ts != (1, 1, TILE, TILE):
        raise ValueError(f"{OP_NAME}: trans_mat must be [1, 1, {TILE}, {TILE}], got {list(ts)}")


def _check_companions(x, others):
    """R8: cos, sin and T are placed like the input (same buffer, sharding and shard cores) and share its dtype
    and layout."""
    x_axes = memory.memory_axes(x)
    x_grid = x.memory_config().shard_spec.grid if x.memory_config().shard_spec is not None else None
    for name, tensor in others.items():
        if tensor.dtype != x.dtype or tensor.layout != x.layout:
            raise NotImplementedError(
                f"{OP_NAME}: {name} is {tensor.dtype}/{tensor.layout}, input_tensor is {x.dtype}/{x.layout}: "
                "the four tensors must share one dtype and one layout"
            )
        if memory.memory_axes(tensor) != x_axes:
            raise NotImplementedError(
                f"{OP_NAME}: {name} is placed {memory.memory_axes(tensor)}, input_tensor {x_axes}: "
                "the four tensors must be placed alike"
            )
        spec = tensor.memory_config().shard_spec
        if spec is None or spec.grid != x_grid:
            raise NotImplementedError(f"{OP_NAME}: {name} must be sharded on input_tensor's cores {x_grid}")


def validate(input_tensor, cos_cache, sin_cache, trans_mat, *, is_decode_mode=False, compute_kernel_config=None):
    _check_shapes(input_tensor, cos_cache, sin_cache, trans_mat)
    config = _resolve_compute_config(compute_kernel_config)
    axes = {
        "dtype":            input_tensor.dtype,
        "layout":           input_tensor.layout,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "is_decode_mode":   bool(is_decode_mode),
        **memory.memory_axes(input_tensor),
    }
    shapes = tuple(_shape(t) for t in (input_tensor, cos_cache, sin_cache, trans_mat))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(shapes, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{OP_NAME}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"{OP_NAME}: unsupported combination (refinement candidate): {exc}")

    _check_companions(input_tensor, {"cos_cache": cos_cache, "sin_cache": sin_cache, "trans_mat": trans_mat})
    return config


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rotary_embedding_llama(
    input_tensor: ttnn.Tensor,
    cos_cache: ttnn.Tensor,
    sin_cache: ttnn.Tensor,
    trans_mat: ttnn.Tensor,
    *,
    is_decode_mode: bool = False,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    config = validate(
        input_tensor, cos_cache, sin_cache, trans_mat,
        is_decode_mode=is_decode_mode, compute_kernel_config=compute_kernel_config,
    )
    return _dispatch(input_tensor, cos_cache, sin_cache, trans_mat, config)


def _dispatch(input_tensor, cos_cache, sin_cache, trans_mat, config, **artifact_options):
    """One generic_op dispatch. `artifact_options` reach create_program_artifacts (the thread plan and the test
    probes); the public entry point passes none."""
    # R9: the result has the input tensor's memory config, shard spec included.
    output = ttnn.allocate_tensor_on_device(input_tensor.spec, input_tensor.device())
    tensors = [input_tensor, cos_cache, sin_cache, trans_mat, output]
    spec, run_args, tensor_indices = create_program_artifacts(*tensors, config, **artifact_options)
    return ttnn.generic_op(tensors, spec, run_args, tensor_indices)
