# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""genify_matmul: out = epilogue(A @ B) as ONE fused device program (one ttnn.generic_op dispatch).

Registry model (see eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS and validate() are
declared inline; the public entry point runs the contract checks (ValueError), then validate()
(NotImplementedError subclasses), then builds and dispatches the program descriptor.

Phase 0 builds regime R1 `interleaved_out_split` (op_design.md, Blocking Model): an independent
M x N output split over the device grid, each core walking its own region in blocks with a K-walk
accumulated in L1 (packer L1 accumulation), epilogue none.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .genify_matmul_program_descriptor import TILE, create_program_descriptor

_ML = ttnn.TensorMemoryLayout


# ---------------------------------------------------------------------------
# Default compute config (single factory; the golden axis tagger reads it too)
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """HiFi4, fp32_dest_acc_en=True, math_approx_mode=False."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS  (inputs = (A_shape, B_shape))
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """'tile_aligned' when K, N, M are all multiples of 32; else the first non-aligned in K, N, M."""
    a_shape, b_shape = inputs[0], inputs[1]
    k, n, m = a_shape[-1], b_shape[-1], a_shape[-2]
    if k % TILE:
        return "k_non_aligned"
    if n % TILE:
        return "n_non_aligned"
    if m % TILE:
        return "m_non_aligned"
    return "tile_aligned"


def tag_weight_batch(inputs, axes):
    """'batched' when B has a leading dim > 1, else 'single'."""
    b_shape = inputs[1]
    return "batched" if any(d > 1 for d in list(b_shape)[:-2]) else "single"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "weight_batch": tag_weight_batch,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED  (every axis of feature_spec TARGET)
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "weight_dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "fp32_dest_acc_en": [True],
    "placement": ["interleaved"],
    "epilogue": ["none"],
    # m_non_aligned is a knob-turn already realized by the Phase 0 geometry (op_design.md, TARGET
    # table): M is tiled with ceil per image, every image is tile-padded on its own, and padded A rows
    # only feed padded output rows. K / N non-aligned are outside TARGET.
    "alignment": ["tile_aligned", "m_non_aligned"],
    "weight_batch": ["single"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # Precision convention (task rule): fp32 activations need the fp32 DEST accumulator. Permanent.
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # dram_sharded with a batched weight is used by no model (task rule allows it). Permanent.
    {"placement": "dram_sharded", "weight_batch": "batched"},
]


# ---------------------------------------------------------------------------
# Axis classification (mirrors eval/golden_tests/genify_matmul/axes.py)
# ---------------------------------------------------------------------------


def _classify_placement(input_tensor, weight):
    w = weight.memory_config()
    if w.buffer_type == ttnn.BufferType.DRAM and w.memory_layout == _ML.WIDTH_SHARDED:
        return "dram_sharded"
    return {
        _ML.INTERLEAVED: "interleaved",
        _ML.HEIGHT_SHARDED: "height_sharded",
        _ML.BLOCK_SHARDED: "block_sharded",
    }.get(input_tensor.memory_config().memory_layout, "other")


def _classify_epilogue(bias, activation, residual, gate):
    if residual is not None or gate is not None:
        return "addcmul" if bias is None and activation is None else "other"
    if activation == "swiglu":
        return "swiglu" if bias is None else "other"
    if activation == "relu":
        return "bias_relu" if bias is not None else "other"
    if activation is None:
        return "bias" if bias is not None else "none"
    return "other"


def _resolve_compute_config(compute_kernel_config):
    return compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(
    input_tensor,
    weight,
    *,
    bias=None,
    activation=None,
    residual=None,
    gate=None,
    compute_kernel_config=None,
    memory_config=None,
):
    """Runtime support gate: SUPPORTED per axis, then EXCLUSIONS (both NotImplementedError)."""
    cfg = _resolve_compute_config(compute_kernel_config)
    axes = {
        "dtype": input_tensor.dtype,
        "weight_dtype": weight.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "placement": _classify_placement(input_tensor, weight),
        "epilogue": _classify_epilogue(bias, activation, residual, gate),
    }
    inputs = (list(input_tensor.shape), list(weight.shape))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"genify_matmul: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"genify_matmul: unsupported combination (refinement candidate): {exc}")

    # Phase 0 (R1) only reads B and writes the output through interleaved accessors.
    if weight.memory_config().memory_layout != _ML.INTERLEAVED:
        raise UnsupportedAxisValue("genify_matmul: placement 'interleaved' needs an interleaved weight")
    if weight.layout != ttnn.TILE_LAYOUT:
        raise UnsupportedAxisValue("genify_matmul: weight layout must be TILE_LAYOUT")
    if memory_config is not None and memory_config.memory_layout != _ML.INTERLEAVED:
        raise UnsupportedAxisValue(
            f"genify_matmul: placement 'interleaved' writes an interleaved output; got {memory_config.memory_layout}"
        )
    return axes


# ---------------------------------------------------------------------------
# Contract checks (ValueError, before the support gate)
# ---------------------------------------------------------------------------


def _is_row_vector(shape, n):
    return len(shape) >= 2 and shape[-1] == n and all(d == 1 for d in shape[:-1])


def _check_contract(input_tensor, weight, bias, activation, residual, gate):
    a_shape = list(input_tensor.shape)
    b_shape = list(weight.shape)
    if len(a_shape) < 2:
        raise ValueError(f"genify_matmul: A must have rank >= 2, got shape {a_shape}")
    if len(b_shape) < 2:
        raise ValueError(f"genify_matmul: B must have rank >= 2, got shape {b_shape}")
    if a_shape[-1] != b_shape[-2]:
        raise ValueError(f"genify_matmul: A[-1]={a_shape[-1]} != B[-2]={b_shape[-2]}")
    # B is one shared (K, N) matrix (all leading dims 1), or batched with leading dims equal to A's
    # (B's leading dims left-padded with 1s to A's rank, as broadcasting would).
    a_lead, b_lead = a_shape[:-2], b_shape[:-2]
    b_lead_padded = [1] * (len(a_lead) - len(b_lead)) + b_lead
    shared_b = all(d == 1 for d in b_lead)
    if len(b_shape) > 2 and not (len(b_lead) <= len(a_lead) and (shared_b or b_lead_padded == a_lead)):
        raise ValueError(
            f"genify_matmul: B must be 2-D or batched with leading dims equal to A's; A {a_shape}, B {b_shape}"
        )
    if activation not in (None, "relu", "swiglu"):
        raise ValueError(f"genify_matmul: activation must be None, 'relu' or 'swiglu', got {activation!r}")
    n = b_shape[-1]
    if activation == "swiglu" and n % 64 != 0:
        raise ValueError(f"genify_matmul: swiglu needs N % 64 == 0, got N={n}")
    if (residual is None) != (gate is None):
        raise ValueError("genify_matmul: residual and gate must be given together (addcmul)")
    if bias is not None and not _is_row_vector(list(bias.shape), n):
        raise ValueError(f"genify_matmul: bias must be (1, {n}), got {list(bias.shape)}")
    if gate is not None and not _is_row_vector(list(gate.shape), n):
        raise ValueError(f"genify_matmul: gate must be (1, {n}), got {list(gate.shape)}")
    if residual is not None and list(residual.shape) != a_shape[:-1] + [n]:
        raise ValueError(f"genify_matmul: residual must be {a_shape[:-1] + [n]}, got {list(residual.shape)}")
    if _classify_epilogue(bias, activation, residual, gate) == "other":
        raise ValueError(
            "genify_matmul: unsupported kwarg combination (relu needs bias; swiglu takes no bias; "
            "addcmul takes no bias/activation)"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def genify_matmul(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    bias: ttnn.Tensor = None,
    activation: str = None,
    residual: ttnn.Tensor = None,
    gate: ttnn.Tensor = None,
    dtype: ttnn.DataType = None,
    memory_config: ttnn.MemoryConfig = None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    """out = epilogue(A @ B) in one device program. See op_design.md."""
    _check_contract(input_tensor, weight, bias, activation, residual, gate)
    validate(
        input_tensor,
        weight,
        bias=bias,
        activation=activation,
        residual=residual,
        gate=gate,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )

    cfg = _resolve_compute_config(compute_kernel_config)
    device = input_tensor.device()
    out_dtype = dtype if dtype is not None else input_tensor.dtype
    out_mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    a_shape = list(input_tensor.shape)
    n_out = weight.shape[-1] // 2 if activation == "swiglu" else weight.shape[-1]
    out_shape = a_shape[:-1] + [n_out]

    # allocate_tensor_on_device takes positional args only.
    output_tensor = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), out_dtype, ttnn.TILE_LAYOUT, device, out_mc)

    program_descriptor = create_program_descriptor(input_tensor, weight, output_tensor, cfg)
    # Exactly one dispatch; the output tensor is last.
    return ttnn.generic_op([input_tensor, weight, output_tensor], program_descriptor)
