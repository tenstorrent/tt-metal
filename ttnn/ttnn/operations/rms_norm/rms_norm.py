# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm for Quasar: one Metal 2.0 `ttnn.ProgramSpec` dispatch.

    out[..., j] = x[..., j] * gamma[j] / sqrt(epsilon + mean_k(x[..., k] ** 2)),  gamma = weight.reshape(-1)[:W]

Specification: eval/specs/rms_norm/spec.py. Design: op_design.md (regime `row_resident`: tile-rows dealt to
clusters in whole rounds of C_eff and to the NEOs of a cluster, each NEO holding its tile-row resident across
the reduction pass and the normalisation pass). Host program: rms_norm_program_artifacts.py.
"""

from __future__ import annotations

import ttnn

from eval.memory import memory_axes
from eval.specs.rms_norm.spec import AXES
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rms_norm_program_artifacts import TILE, create_program_artifacts, select_regime

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase (the specification's PHASE0), axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":            [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout":           [ttnn.TILE_LAYOUT],
    "memory_config":    ["DRAM", "L1"],
    "sharding":         ["interleaved"],
    "gamma_dtype":      [ttnn.bfloat16],
    "gamma_layout":     [ttnn.ROW_MAJOR_LAYOUT],
    "memory_argument":  [None],
    "alignment":        ["tile_aligned", "h_non_aligned"],
    "rank":             [4],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: cells inside cartesian(SUPPORTED) refused for now. None.
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# Rule R1 of the specification: float32 with the 16-bit accumulator is legal but lossy, and refused in every
# phase whatever SUPPORTED says.
_R1_REFUSED = {"dtype": ttnn.float32, "fp32_dest_acc_en": False}


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means (rule R2): the one place the default is defined."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _resolve_compute_config(compute_kernel_config):
    """A `ttnn.ComputeConfigDescriptor` for the kernel (rule R4).

    Takes a descriptor or tt-metal's device compute kernel config, reads the three fields as attributes and
    ignores the ones a descriptor does not have (`packer_l1_acc`, `dst_full_sync_en`, `throttle_level`).
    """
    config = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    if isinstance(config, ttnn.ComputeConfigDescriptor):
        return config
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=config.math_fidelity,
        math_approx_mode=bool(config.math_approx_mode),
        fp32_dest_acc_en=bool(config.fp32_dest_acc_en),
    )


def _check_weight_form(input_tensor, weight):
    """Rejection X1: the weight must hold one gamma per column, 32 to a row, ceil(W / 32) rows."""
    width = int(input_tensor.shape[-1])
    wshape = tuple(int(d) for d in weight.shape)
    numel = 1
    for d in wshape:
        numel *= d
    expected = -(-width // TILE) * TILE
    if len(wshape) == 0 or wshape[-1] != TILE or numel != expected:
        raise ValueError(
            f"rms_norm: weight of shape {list(wshape)} does not hold one gamma per column of a W={width} input: "
            f"it needs a last dimension of {TILE} and {expected} elements ([1, 1, {expected // TILE}, {TILE}])"
        )


def validate(
    input_tensor,
    *,
    epsilon=1e-12,
    weight,
    memory_config=None,
    program_config=None,
    compute_kernel_config=None,
):
    """The runtime gate: X1, then the axes of the call against SUPPORTED / EXCLUSIONS (and rule R1), then the
    regime predicate. `program_config` is never read (rule R14); neither are math_fidelity and
    math_approx_mode (rule R3)."""
    _check_weight_form(input_tensor, weight)
    config = _resolve_compute_config(compute_kernel_config)
    inputs = (tuple(int(d) for d in input_tensor.shape), tuple(int(d) for d in weight.shape), epsilon)
    axes = {
        "dtype":            input_tensor.dtype,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "layout":           input_tensor.layout,
        **memory_axes(input_tensor),                        # memory_config, sharding (rule R12)
        "gamma_dtype":      weight.dtype,                   # rule R8
        "gamma_layout":     weight.layout,
        "memory_argument":  None if memory_config is None else "other",  # rule R13
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    if all(axes.get(k) == v for k, v in _R1_REFUSED.items()):
        raise UnsupportedAxisValue("rms_norm: float32 input with fp32_dest_acc_en=False is refused (rule R1)")
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"rms_norm: unsupported combination (refinement candidate): {exc}")

    # The regime predicate: a tile-row must fit one ring at one NEO (row_resident); raises otherwise.
    select_regime(input_tensor, config)
    return config


def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    epsilon: float = 1e-12,
    weight: ttnn.Tensor,
    memory_config: "ttnn.MemoryConfig" = None,
    program_config=None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    """Root-mean-square normalisation over the last dimension, scaled by gamma = weight.reshape(-1)[:W]."""
    config = validate(
        input_tensor,
        epsilon=epsilon,
        weight=weight,
        memory_config=memory_config,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    # Rule R11: the output takes the input's spec (shape, dtype, layout and memory config).
    output = ttnn.allocate_tensor_on_device(input_tensor.spec, input_tensor.device())
    spec, run_args, tensor_indices = create_program_artifacts(
        input_tensor, weight, output, epsilon=float(epsilon), compute_config=config
    )
    return ttnn.generic_op([input_tensor, weight, output], spec, run_args, tensor_indices)
