# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""max_pool2d: 2-D max pooling of N channels-last images given flattened as [1, 1, N*H*W, C].

Registry model (eval/REGISTRY_MODEL.md): INPUT_TAGGERS (the specification's AXES), SUPPORTED, EXCLUSIONS and
validate(), then the public entry point, which validates first and then dispatches exactly one
`ttnn.generic_op` (max_pool2d_program_artifacts.py builds the Metal 2.0 ProgramSpec).

Specification: eval/specs/max_pool2d/spec.py. Design: op_design.md in this directory.
"""

from __future__ import annotations

import ttnn

from eval import memory
from eval.specs.max_pool2d.spec import AXES
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .max_pool2d_program_artifacts import Geometry, create_program_artifacts, plan_output

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase (= the specification's PHASE0, axis for axis).
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "memory_config": ["DRAM", "L1"],
    "sharding": ["interleaved", "height", "width"],
    "batch": ["single"],
    "channel_alignment": ["tile_aligned"],
    "window": ["overlapping"],
    "pad": ["padded"],
    "dilate": ["none"],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: none inside the rectangle.
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means (rule R3): the one place the default is defined."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _pair(name, value, *, minimum):
    """A window argument as a tuple (h, w) of ints (rule R5: a list or a tuple of two ints). Anything else is
    malformed (X4), as is a value below `minimum`."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"max_pool2d: {name} must be a list or tuple of two ints, got {value!r}")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise ValueError(f"max_pool2d: {name} must hold two ints, got {value!r}")
    if any(v < minimum for v in value):
        raise ValueError(f"max_pool2d: every value of {name} must be at least {minimum}, got {value!r}")
    return int(value[0]), int(value[1])


def _canonical_call(input_tensor, batch_size, input_h, input_w, channels, kernel_size, stride, padding, dilation):
    """The call's arguments in canonical form, after the malformed-call checks (X1-X4). Returns the Geometry
    and the INPUTS-shaped entry the axis functions read."""
    kernel = _pair("kernel_size", kernel_size, minimum=1)
    strides = _pair("stride", stride, minimum=1)
    pads = _pair("padding", padding, minimum=0)
    dilations = _pair("dilation", dilation, minimum=1)
    for name, v in (("batch_size", batch_size), ("input_h", input_h), ("input_w", input_w), ("channels", channels)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"max_pool2d: {name} must be a positive int, got {v!r}")

    shape = tuple(int(d) for d in input_tensor.shape)
    expected = (1, 1, batch_size * input_h * input_w, channels)
    if shape != expected:  # X2
        raise ValueError(
            f"max_pool2d: the input must be [1, 1, batch_size*input_h*input_w, channels] = {list(expected)}, "
            f"got {list(shape)}")
    for i, dim in enumerate("hw"):  # X1: torch's limit, padding at most half the kernel
        if 2 * pads[i] > kernel[i]:
            raise ValueError(f"max_pool2d: padding {pads} is more than half the kernel size {kernel} along {dim}")
    geo = Geometry.of(batch_size, input_h, input_w, channels, kernel, strides, pads, dilations)
    if geo.H_out < 1 or geo.W_out < 1:  # X3
        raise ValueError(
            f"max_pool2d: the dilated kernel {kernel} (dilation {dilations}) does not fit the padded "
            f"{input_h} x {input_w} image")
    entry = (shape, batch_size, input_h, input_w, channels, kernel, strides, pads, dilations)
    return geo, entry


def _axes(input_tensor, entry, config):
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        **memory.memory_axes(input_tensor),  # rule R8
    }
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger(entry, axes)
    return axes


def validate(input_tensor, batch_size, input_h, input_w, channels, kernel_size, stride, padding, dilation, *,
             compute_kernel_config=None):
    """Malformed calls raise ValueError (REJECTIONS X1-X4); a well-formed call outside SUPPORTED or inside
    EXCLUSIONS raises the pipeline's refusal (a NotImplementedError, rule R12). Neither `math_fidelity` nor
    `math_approx_mode` is read (rule R4). Returns the canonical Geometry."""
    config = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    geo, entry = _canonical_call(input_tensor, batch_size, input_h, input_w, channels, kernel_size, stride, padding,
                                 dilation)
    axes = _axes(input_tensor, entry, config)

    # Rule R2: float32 with a 16-bit DEST is legal but lossy, and refused natively.
    if axes["dtype"] == ttnn.float32 and not axes["fp32_dest_acc_en"]:
        raise UnsupportedAxisValue("max_pool2d: float32 input requires fp32_dest_acc_en=True")
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"max_pool2d: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"max_pool2d: unsupported combination (refinement candidate): {exc}")
    return geo


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def max_pool2d(
    input_tensor: ttnn.Tensor,
    batch_size: int,
    input_h: int,
    input_w: int,
    channels: int,
    kernel_size,
    stride,
    padding,
    dilation,
    *,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    """torch.nn.functional.max_pool2d on the images [N, C, H, W] read out of `input_tensor` = [1, 1, N*H*W, C],
    returned flattened the same way as [1, 1, N*H_out*W_out, C]: row-major, the input's dtype, sharded in L1
    (rules R9, R11). One device-program dispatch."""
    geo = validate(input_tensor, batch_size, input_h, input_w, channels, kernel_size, stride, padding, dilation,
                   compute_kernel_config=compute_kernel_config)
    config = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    device = input_tensor.device()
    out_plan = plan_output(input_tensor, geo, device)
    output = ttnn.allocate_tensor_on_device(out_plan.spec, device)
    spec, run_args, tensor_indices = create_program_artifacts(input_tensor, output, geo, out_plan, config)
    return ttnn.generic_op([input_tensor, output], spec, run_args, tensor_indices)
