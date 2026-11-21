# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""sigmoid(x) = 1 / (1 + exp(-x)), element by element, on Quasar.

One `ttnn.generic_op` dispatch of one `ttnn.ProgramSpec` (sigmoid_program_artifacts.py): every cluster of the
grid streams a contiguous run of tile pages through a thread plan of reader threads, NEO compute threads and
writer threads. The compute kernel is one fused eltwise chain per NEO (copy -> fp32-accurate sigmoid -> pack),
so the whole expression stays in DEST.

Specification: eval/specs/sigmoid/spec.py. Design: op_design.md beside this file.
"""

from __future__ import annotations

import ttnn

from eval.memory import memory_axes
from eval.specs.sigmoid.spec import AXES
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .sigmoid_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase is the specification's PHASE0, axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":            [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout":           [ttnn.TILE_LAYOUT],
    "memory_config":    ["DRAM", "L1"],
    "sharding":         ["interleaved"],
    "alignment":        ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank":             [2, 3, 4],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: none in the first phase. (R1, float32 with fp32_dest_acc_en=False, is outside SUPPORTED
#    today; it becomes a permanent entry here when float32 joins.)
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# Compute config default (R2): defined once, here, and exported.
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _resolve_config(compute_kernel_config):
    return compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()


# ---------------------------------------------------------------------------
# 4. validate(): the runtime gate (R5, R7). Reads neither math_fidelity nor math_approx_mode (R3).
# ---------------------------------------------------------------------------


def validate(input_tensor, *, compute_kernel_config=None):
    config = _resolve_config(compute_kernel_config)
    axes = {
        "dtype":            input_tensor.dtype,
        "layout":           input_tensor.layout,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        **memory_axes(input_tensor),
    }
    inputs = (tuple(input_tensor.shape),)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"sigmoid: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"sigmoid: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _sigmoid(input_tensor, compute_kernel_config, **artifact_options):
    """The dispatch. `artifact_options` reach create_program_artifacts (thread plan, test probes); the public
    entry point passes none, so callers always get the design's plan."""
    validate(input_tensor, compute_kernel_config=compute_kernel_config)
    config = _resolve_config(compute_kernel_config)
    # R6: the output lives in the input's buffer, interleaved; the spec carries the input's memory config.
    output = ttnn.allocate_tensor_on_device(input_tensor.spec, input_tensor.device())
    spec, run_args, tensor_indices = create_program_artifacts(input_tensor, output, config, **artifact_options)
    return ttnn.generic_op([input_tensor, output], spec, run_args, tensor_indices)


def sigmoid(
    input_tensor: ttnn.Tensor,
    *,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    """torch.sigmoid(input_tensor), as one device program."""
    return _sigmoid(input_tensor, compute_kernel_config)
