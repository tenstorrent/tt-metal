# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""softmax: numerically stable softmax along one dimension, as one Metal 2.0 ProgramSpec dispatch on Quasar.

The specification is eval/specs/softmax/spec.py; the design is op_design.md beside this file.
"""

from __future__ import annotations

import ttnn

from eval.memory import memory_axes
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .softmax_program_artifacts import create_program_artifacts, select_regime

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

from eval.specs.softmax.spec import AXES  # noqa: E402

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: equals the specification's PHASE0 after the first phase.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout": [ttnn.TILE_LAYOUT],
    "memory_config": ["DRAM", "L1"],
    "sharding": ["interleaved"],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [4],
    "dim": [-2],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: none inside the first-phase rectangle.
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# Canonical axes softmax can ever reduce along (rejection X1 refuses anything else).
_REDUCIBLE_DIMS = (-1, -2)


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means (rule R4): defined here only."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )


def resolve_compute_kernel_config(compute_kernel_config):
    """A ttnn.ComputeConfigDescriptor for the kernel (rules R4, R10).

    None resolves through default_compute_kernel_config(). Anything else, a descriptor or tt-metal's device
    config (WormholeComputeKernelConfig), is read by attribute, never by type; fields a descriptor does not
    have (packer_l1_acc, dst_full_sync_en, throttle_level) are ignored.
    """
    config = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=config.math_fidelity,
        math_approx_mode=bool(config.math_approx_mode),
        fp32_dest_acc_en=bool(config.fp32_dest_acc_en),
    )


def canonical_dim(dim, ndim):
    """Rule R2: a negative offset, Torch convention."""
    return dim if dim < 0 else dim - ndim


def validate(input_tensor, dim=-1, *, compute_kernel_config=None):
    shape = tuple(input_tensor.shape)
    ndim = len(shape)
    if not isinstance(dim, int) or not (-ndim <= dim < ndim):
        raise ValueError(f"softmax: dim={dim!r} is out of range for a rank-{ndim} tensor")
    cdim = canonical_dim(dim, ndim)
    if cdim not in _REDUCIBLE_DIMS:
        raise ValueError(f"softmax: dim={dim} canonicalises to {cdim}; only -1 and -2 are reducible axes")

    config = resolve_compute_kernel_config(compute_kernel_config)
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "dim": cdim,
        **memory_axes(input_tensor),  # rule R9: memory_config, sharding
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"softmax: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"softmax: unsupported combination (refinement candidate): {exc}")

    # Rule R3 (float32 with a 16-bit accumulator is legal but lossy): refused natively.
    if axes["dtype"] == ttnn.float32 and not axes["fp32_dest_acc_en"]:
        raise UnsupportedAxisValue("softmax: float32 input with fp32_dest_acc_en=False is refused (rule R3)")

    # The built regime keeps a whole column resident: refuse a column too tall for one ring.
    select_regime(input_tensor)
    return axes


def softmax(
    input_tensor: ttnn.Tensor,
    dim: int = -1,
    *,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    validate(input_tensor, dim, compute_kernel_config=compute_kernel_config)
    config = resolve_compute_kernel_config(compute_kernel_config)
    # Rule R8: the output lives in the input's buffer (the spec carries the input's memory config).
    output = ttnn.allocate_tensor_on_device(input_tensor.spec, input_tensor.device())
    spec, run_args, tensor_indices = create_program_artifacts(input_tensor, output, config)
    return ttnn.generic_op([input_tensor, output], spec, run_args, tensor_indices)
