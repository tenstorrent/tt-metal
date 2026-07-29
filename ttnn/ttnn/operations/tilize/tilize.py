# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""``tilize`` — re-lay a ROW_MAJOR tensor into TILE layout on device.

Pure layout conversion: element VALUES are unchanged, only byte positions move.
An optional ``dtype=`` narrows/widens the storage format of the result.

Structure follows ``eval/op_template.py``:

1. ``INPUT_TAGGERS`` — project the golden-suite scenario dict onto categorical axes.
2. ``SUPPORTED``     — per-axis accepted values.
3. ``EXCLUSIONS``    — cells inside cartesian(SUPPORTED) refused for now.
4. ``validate()``    — runtime gate, called first by the public entry point.

``INVALID`` is deliberately absent — it lives in
``eval/golden_tests/tilize/feature_spec.py``.
"""

from __future__ import annotations

from math import prod

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .tilize_program_descriptor import build_plan, create_program_descriptor

TILE_HW = 32

_HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
_BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# Each tagger reads ``inputs[0]`` — the golden suite's scenario dict (see
# eval/golden_tests/tilize/feature_spec.py). ``validate()`` derives the same
# axis names straight off the real call arguments instead (a live ttnn.Tensor
# is not a scenario dict), so the two paths agree on axis *names* and *values*
# without sharing a code path.

_BUFFER_NAME = {ttnn.BufferType.DRAM: "dram", ttnn.BufferType.L1: "l1"}


def _spec_out_scheme(spec):
    """ "interleaved" for an interleaved spec, else its scheme ("nd" when None)."""
    if spec.get("kind") == "interleaved":
        return "interleaved"
    scheme = spec.get("scheme")
    return "nd" if scheme is None else scheme


def tag_use_multicore(inputs, axes):
    return bool(inputs[0]["use_multicore"])


def tag_shard_api(inputs, axes):
    return inputs[0]["shard_api"]


def tag_out_scheme(inputs, axes):
    return _spec_out_scheme(inputs[0]["out"])


def tag_buffer(inputs, axes):
    scenario = inputs[0]
    return f"{_BUFFER_NAME[scenario['in']['buffer']]}_to_{_BUFFER_NAME[scenario['out']['buffer']]}"


def tag_rank(inputs, axes):
    return int(len(inputs[0]["input_shape"]))


def tag_double_buffer(inputs, axes):
    return bool(inputs[0].get("use_double_buffer", True))


INPUT_TAGGERS = {
    "use_multicore": tag_use_multicore,
    "shard_api": tag_shard_api,
    "out_scheme": tag_out_scheme,
    "buffer": tag_buffer,
    "rank": tag_rank,
    "double_buffer": tag_double_buffer,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
#
# ``dtype`` also lists uint16/int32 (not in the golden TARGET but exercised by
# the acceptance test's integer-passthrough cases). Extra values beyond TARGET
# are harmless — the harness only checks the values a cell actually takes.

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.uint32, ttnn.uint16, ttnn.int32],
    "output_dtype": [
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.bfloat8_b,
        ttnn.uint32,
        ttnn.uint16,
        ttnn.int32,
    ],
    "use_multicore": [False, True],
    "double_buffer": [False, True],
    "shard_api": ["none", "legacy_2d", "nd"],
    "out_scheme": ["interleaved", _HEIGHT, _WIDTH, _BLOCK, "nd"],
    "buffer": ["dram_to_dram", "dram_to_l1", "l1_to_l1", "l1_to_dram"],
    "rank": [2, 3, 4],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# Empty. op_design.md proposed excluding {use_multicore: False} x sharded on the
# grounds that "sharded I/O is inherently multi-core". That turned out not to be
# a kernel-level boundary here: the generic (TensorAccessor) path addresses
# sharded pages from any core count, so `use_multicore=False` with sharded I/O
# simply routes to a 1-core generic program instead of the zero-copy Path B.
# Declaring it excluded would have been a refusal the op does not actually need
# — and the reference suite exercises exactly that cell.

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    "multi_core": {"value": True, "source": "verified"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _is_nd_sharded(memory_config) -> bool:
    if not memory_config.is_sharded():
        return False
    if memory_config.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED:
        return True
    return memory_config.shard_spec is None and memory_config.nd_shard_spec is not None


def _out_scheme_axis(memory_config):
    if not memory_config.is_sharded():
        return "interleaved"
    if _is_nd_sharded(memory_config):
        return "nd"
    return memory_config.memory_layout


def _shard_api_axis(in_memory_config, out_memory_config) -> str:
    if not (in_memory_config.is_sharded() or out_memory_config.is_sharded()):
        return "none"
    if _is_nd_sharded(in_memory_config) or _is_nd_sharded(out_memory_config):
        return "nd"
    return "legacy_2d"


def validate(
    input_tensor,
    out_memory_config,
    *,
    output_dtype,
    use_multicore=True,
    use_double_buffer=True,
) -> None:
    """Runtime support gate. SUPPORTED first (per-axis), then EXCLUSIONS (cell)."""
    in_memory_config = input_tensor.memory_config()

    axes = {
        "dtype": input_tensor.dtype,
        "output_dtype": output_dtype,
        "use_multicore": bool(use_multicore),
        "double_buffer": bool(use_double_buffer),
        "shard_api": _shard_api_axis(in_memory_config, out_memory_config),
        "out_scheme": _out_scheme_axis(out_memory_config),
        "buffer": (
            f"{_BUFFER_NAME[in_memory_config.buffer_type]}_to_" f"{_BUFFER_NAME[out_memory_config.buffer_type]}"
        ),
        "rank": int(len(input_tensor.shape)),
    }

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"tilize: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside SUPPORTED
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"tilize: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Structural / shape validation (runs BEFORE validate(), per the design)
# ---------------------------------------------------------------------------


def _check_structure(input_tensor) -> None:
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise RuntimeError(f"tilize requires ROW_MAJOR_LAYOUT input, got {input_tensor.layout}")

    try:
        device = input_tensor.device()
    except Exception:  # noqa: BLE001 - host tensors raise, that is the signal
        device = None
    if device is None:
        raise RuntimeError("tilize requires a tensor on device (got a host tensor)")

    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise RuntimeError(f"tilize requires rank >= 2, got rank {len(shape)} ({shape})")

    if shape[-1] % TILE_HW != 0 or shape[-2] % TILE_HW != 0:
        raise ValueError(
            f"tilize: last two dims must be divisible by 32, got {shape[-2:]} "
            "(this op does not pad — use tilize_with_val_padding)"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def tilize(
    input_tensor: ttnn.Tensor,
    memory_config: ttnn.MemoryConfig = None,
    *,
    dtype: ttnn.DataType = None,
    use_multicore: bool = True,
    use_double_buffer: bool = True,
) -> ttnn.Tensor:
    """Convert ``input_tensor`` from ROW_MAJOR to TILE layout.

    Args:
        input_tensor: ROW_MAJOR tensor on device, rank >= 2, last two dims % 32 == 0.
        memory_config: output memory config; defaults to the input's.
        dtype: output dtype; defaults to the input's (value-preserving cast).
        use_multicore: distribute the work over the compute grid (default True).
        use_double_buffer: depth-2 circular buffers (default True).

    Returns:
        A TILE_LAYOUT tensor with the same logical shape and values.
    """
    _check_structure(input_tensor)

    out_memory_config = memory_config if memory_config is not None else input_tensor.memory_config()
    out_dtype = dtype if dtype is not None else input_tensor.dtype

    validate(
        input_tensor,
        out_memory_config,
        output_dtype=out_dtype,
        use_multicore=use_multicore,
        use_double_buffer=use_double_buffer,
    )

    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        out_dtype,
        ttnn.TILE_LAYOUT,
        device,
        out_memory_config,
    )

    plan = build_plan(
        input_tensor,
        output_tensor,
        device,
        use_multicore=use_multicore,
        use_double_buffer=use_double_buffer,
    )
    program_descriptor = create_program_descriptor(input_tensor, output_tensor, plan)

    # Output tensor MUST be last.
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


# Re-exported so callers/tools can introspect the derived geometry without
# building a program.
def folded_geometry(tensor):
    """(folded_H, W, nt_h, Wt) for ``tensor``'s padded shape."""
    padded = list(tensor.padded_shape)
    folded_h = int(prod(padded[:-1]))
    width = int(padded[-1])
    return folded_h, width, folded_h // TILE_HW, width // TILE_HW
