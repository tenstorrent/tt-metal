# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""linear: out = a @ b (+ bias) over the last two dimensions, on Quasar.

The specification is eval/specs/linear/spec.py; the design is op_design.md next to this file. One
`ttnn.generic_op` dispatch per call (regime R-2D-stream): the output is cut into [block_h x block_w] tile
blocks, C blocks with the same rows make a round (one block per NEO of a cluster), whole rounds are dealt
to clusters, and every NEO walks the whole of K into DEST, onto the bias it preloaded there.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval import memory
from eval.specs.linear.spec import (  # noqa: E402  (the harness imports ttnn first)
    AXES,
    canonical_bias,
    canonical_dtype,
    canonical_memory,
    output_shape,
)

from .linear_program_artifacts import create_program_artifacts, plan_blocking

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase is exactly the specification's PHASE0.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":            [ttnn.bfloat16],
    "fp32_dest_acc_en": [True, False],
    "layout":           [ttnn.TILE_LAYOUT],
    "memory_config":    ["DRAM", "L1"],
    "sharding":         ["interleaved", "width"],
    "output_memory":    ["DRAM", "L1/width", "L1/width/auto"],
    "bias":             [None, "row", "vector"],
    "dtype_argument":   ["same"],
    "alignment":        ["tile_aligned"],
    "rank":             [4],
    "weight_form":      ["matrix", "same_rank"],
    "batch":            ["single"],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: none.
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means (rule R2): HiFi4, a float32 accumulator, no approximation."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def resolve_compute_config(compute_kernel_config):
    """The kernel's `ttnn.ComputeConfigDescriptor` (rule R4): built from the three fields of whatever was
    passed (a descriptor or tt-metal's device compute kernel config), None resolving through the default
    factory. The fields a descriptor does not have (packer_l1_acc, dst_full_sync_en, throttle_level) are
    ignored."""
    # Rule R2: the default lives in the factory only; a field the passed config lacks falls back to it.
    default = default_compute_kernel_config()
    config = compute_kernel_config if compute_kernel_config is not None else default
    fidelity = getattr(config, "math_fidelity", default.math_fidelity)
    if fidelity == getattr(ttnn.MathFidelity, "Invalid", None):  # an unset WormholeComputeKernelConfig field
        fidelity = default.math_fidelity
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity,
        fp32_dest_acc_en=bool(getattr(config, "fp32_dest_acc_en", default.fp32_dest_acc_en)),
        math_approx_mode=bool(getattr(config, "math_approx_mode", default.math_approx_mode)),
    )


def _require_dram_tile_operand(name, tensor, dtype):
    """Rule R6: the weight and the bias are interleaved in DRAM, in tile layout, of the activation's dtype."""
    placed = memory.memory_axes(tensor)
    if placed != {"memory_config": "DRAM", "sharding": "interleaved"}:
        raise NotImplementedError(f"linear: {name} must be interleaved in DRAM, it is {placed}")
    if tensor.layout != ttnn.TILE_LAYOUT:
        raise NotImplementedError(f"linear: {name} must be in TILE_LAYOUT, it is {tensor.layout}")
    if tensor.dtype != dtype:
        raise NotImplementedError(f"linear: {name} must have input_tensor_a's dtype {dtype}, it is {tensor.dtype}")


def validate(
    input_tensor_a,
    input_tensor_b,
    *,
    bias=None,
    memory_config=None,
    dtype=None,
    program_config=None,
    compute_kernel_config=None,
    core_grid=None,
):
    """The runtime gate: malformed calls first (X1, X2, X3, X6, X7), then R11, then every axis on
    SUPPORTED and EXCLUSIONS. Returns the cell's axes."""
    a_shape, b_shape = tuple(input_tensor_a.shape), tuple(input_tensor_b.shape)
    # X2: each operand has at least two dimensions.
    if len(a_shape) < 2 or len(b_shape) < 2:
        raise ValueError(f"linear: each operand needs at least 2 dimensions, got {list(a_shape)} and {list(b_shape)}")
    # X1: the inner dimensions agree.
    if a_shape[-1] != b_shape[-2]:
        raise ValueError(f"linear: inner dimensions differ: {list(a_shape)} @ {list(b_shape)}")
    arguments = {"input_tensor_a": input_tensor_a, "input_tensor_b": input_tensor_b}
    # X3 (ValueError inside), and "block" for a full bias.
    bias_axis = canonical_bias(bias, arguments)
    # X6 (TypeError inside) and R8.
    output_memory = canonical_memory(memory_config, arguments)
    # R9.
    dtype_argument = canonical_dtype(dtype, arguments)
    # R11: a fused activation changes the result.
    if getattr(program_config, "fused_activation", None) is not None:
        raise NotImplementedError("linear: a program_config with a fused_activation is not supported")
    config = resolve_compute_config(compute_kernel_config)
    # R1: float32 operands with a 16-bit accumulator are refused.
    if input_tensor_a.dtype == ttnn.float32 and not config.fp32_dest_acc_en:
        raise NotImplementedError("linear: float32 operands need fp32_dest_acc_en=True")
    # R6.
    _require_dram_tile_operand("input_tensor_b", input_tensor_b, input_tensor_a.dtype)
    if bias is not None:
        _require_dram_tile_operand("bias", bias, input_tensor_a.dtype)

    axes = {
        "dtype":            input_tensor_a.dtype,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "layout":           input_tensor_a.layout,
        **memory.memory_axes(input_tensor_a),  # R5: memory_config, sharding
        "output_memory":    output_memory,
        "bias":             bias_axis,
        "dtype_argument":   dtype_argument,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((a_shape, b_shape), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"linear: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"linear: unsupported combination (refinement candidate): {exc}")
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _output_memory_config(output_memory, memory_config, out_shape, device):
    """Rule R7: where the result goes."""
    if output_memory == "DRAM":
        return ttnn.DRAM_MEMORY_CONFIG
    if output_memory == "L1/width":
        return memory_config
    # "L1/width/auto": all rows per shard, whole tiles of columns, the runner's rule (GUIDANCE).
    return memory.place(memory.Sharded("L1", "width"), tuple(out_shape), ttnn.TILE_LAYOUT, device)


def linear(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    bias: ttnn.Tensor = None,
    memory_config: "ttnn.MemoryConfig" = None,
    dtype: ttnn.DataType = None,
    program_config=None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    core_grid=None,
) -> ttnn.Tensor:
    axes = validate(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
        memory_config=memory_config,
        dtype=dtype,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        core_grid=core_grid,
    )
    # program_config and core_grid are hints (R10): the split below does not read them.
    config = resolve_compute_config(compute_kernel_config)
    device = input_tensor_a.device()
    out_shape = output_shape((tuple(input_tensor_a.shape), tuple(input_tensor_b.shape)), axes)
    out_config = _output_memory_config(axes["output_memory"], memory_config, out_shape, device)
    plan = plan_blocking(input_tensor_a, input_tensor_b, config, device, has_bias=bias is not None)
    output = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape(out_shape), input_tensor_a.dtype, ttnn.TILE_LAYOUT, out_config), device
    )
    spec, run_args, tensor_indices = create_program_artifacts(
        input_tensor_a, input_tensor_b, bias, output, config, plan
    )
    io = [input_tensor_a, input_tensor_b] + ([bias] if bias is not None else []) + [output]
    return ttnn.generic_op(io, spec, run_args, tensor_indices)
