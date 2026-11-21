# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""embedding_gen: a table lookup (`torch.nn.functional.embedding`) as one Metal 2.0 program.

Specification: eval/specs/embedding_gen/spec.py. Design: op_design.md (gather-direct: a reader that fetches the
selected table rows straight into the output layout in a ring entry, a writer that stores the entry; no compute).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval import memory
from eval.specs.embedding_gen.spec import AXES, GENERIC, canonical_layout, canonical_memory, type_word

from .embedding_gen_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase is the specification's PHASE0.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":           [ttnn.bfloat16],
    "layout":          [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "memory_config":   ["DRAM"],
    "sharding":        ["interleaved"],
    "output_layout":   [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "output_memory":   [None, "DRAM"],
    "embeddings_type": ["generic", "padded", "binary"],
    "alignment":       ["tile_aligned", "h_non_aligned"],
    "rank":            [2, 4],
    "table_rank":      [4],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# The tile geometry the kernels' face offsets assume (op_design.md, Mechanism caps).
_TILE_SHAPE = [32, 32]


def _shapes(input_tensor, weight):
    return tuple(int(d) for d in input_tensor.shape), tuple(int(d) for d in weight.shape)


def _check_malformed(indices_shape, table_shape, padding_idx, kind):
    """The REJECTIONS of the specification: calls that are not a lookup at all."""
    if len(table_shape) >= 2 and any(d != 1 for d in table_shape[:-2]):
        raise ValueError(f"embedding_gen: the table must be [V, D] or [1, 1, V, D], got {list(table_shape)} (X1)")
    if len(table_shape) < 2:
        raise ValueError(f"embedding_gen: the table must be [V, D] or [1, 1, V, D], got {list(table_shape)}")
    if len(indices_shape) == 4 and (indices_shape[1] != 1 or indices_shape[2] != 1):
        raise ValueError(f"embedding_gen: indices must be [B, 1, 1, S], got {list(indices_shape)} (X2)")
    if kind == "padded" and padding_idx is None:
        raise ValueError("embedding_gen: embeddings_type PADDED needs a padding_idx (X3)")
    if kind == "binary" and padding_idx is None and table_shape[-2] != 2:
        raise ValueError(f"embedding_gen: embeddings_type BINARY needs a table of two rows, got {table_shape[-2]} (X4)")
    if padding_idx is not None:
        if not isinstance(padding_idx, int) or isinstance(padding_idx, bool):
            raise TypeError(f"embedding_gen: padding_idx must be an int or None, got {padding_idx!r}")
        if not 0 <= padding_idx < table_shape[-2]:
            raise ValueError(f"embedding_gen: padding_idx {padding_idx} is no row of a table of {table_shape[-2]} rows")


def cell_axes(input_tensor, weight, *, padding_idx=None, layout=None, embeddings_type=GENERIC, memory_config=None):
    """The registry cell of a call: tensor properties, canonical arguments and every tagger."""
    indices_shape, table_shape = _shapes(input_tensor, weight)
    kind = type_word(embeddings_type)  # TypeError for anything that is not a ttnn.EmbeddingsType
    _check_malformed(indices_shape, table_shape, padding_idx, kind)
    placed = memory.memory_axes(input_tensor)  # R14
    axes = {
        "dtype": weight.dtype,
        "layout": weight.layout,
        "memory_config": placed["memory_config"],
        "sharding": placed["sharding"],
        "output_layout": canonical_layout(layout, {"weight": weight}),  # R9
        "output_memory": canonical_memory(memory_config, {}),  # R13: TypeError for a non-config (X5)
    }
    inputs = (indices_shape, table_shape, padding_idx, embeddings_type)
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger(inputs, axes)
    return axes


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, weight, *, padding_idx=None, layout=None, embeddings_type=GENERIC, memory_config=None):
    axes = cell_axes(input_tensor, weight, padding_idx=padding_idx, layout=layout,
                     embeddings_type=embeddings_type, memory_config=memory_config)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"embedding_gen: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"embedding_gen: unsupported combination (refinement candidate): {exc}")

    # R5: a block-float table has no row-major form.
    if axes["dtype"] == ttnn.bfloat8_b and axes["output_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        raise UnsupportedAxisValue("embedding_gen: a bfloat8_b table has no row-major output (R5)")

    # Facts the kernels rely on that no axis carries.
    if input_tensor.dtype != ttnn.uint32 or input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise UnsupportedAxisValue(
            f"embedding_gen: indices must be uint32 row-major, got {input_tensor.dtype} {input_tensor.layout}")
    if weight.layout == ttnn.TILE_LAYOUT and list(weight.tile.tile_shape) != _TILE_SHAPE:
        raise UnsupportedAxisValue(f"embedding_gen: only 32x32 tiles, got {weight.tile.tile_shape}")
    if memory.memory_axes(weight) != {"memory_config": "DRAM", "sharding": "interleaved"}:
        raise UnsupportedAxisValue("embedding_gen: the table must be in DRAM, interleaved")
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def embedding_gen(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    padding_idx: int = None,
    layout: ttnn.Layout = None,
    embeddings_type: "ttnn.EmbeddingsType" = GENERIC,
    memory_config: "ttnn.MemoryConfig" = None,
) -> ttnn.Tensor:
    """Row `input_tensor[b, ..., s]` of `weight` for every index: `[B, S, D]` (rules R1-R14).

    `padding_idx` and `embeddings_type` change no result (R6, R7); `layout` is the output's layout (the table's
    when None); `memory_config` is where the output goes (where the indices are when None)."""
    axes = validate(input_tensor, weight, padding_idx=padding_idx, layout=layout,
                    embeddings_type=embeddings_type, memory_config=memory_config)
    return run(input_tensor, weight, out_tiled=axes["output_layout"] == ttnn.TILE_LAYOUT,
               memory_config=memory_config)


def output_shape(indices_shape, table_shape):
    """[B, S, D] for indices [B, S] or [B, 1, 1, S]; [S, D] for indices [S] (R10)."""
    lead = [indices_shape[-1]] if len(indices_shape) == 1 else [indices_shape[0], indices_shape[-1]]
    return lead + [table_shape[-1]]


def run(input_tensor, weight, *, out_tiled, memory_config=None, **plan_overrides):
    """Allocate the output and dispatch the one program. `plan_overrides` reach plan_blocks (tests force the
    ring-cap path with `dfb_max_bytes`)."""
    indices_shape, table_shape = _shapes(input_tensor, weight)
    out_config = memory_config if memory_config is not None else input_tensor.memory_config()  # R12
    out_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, out_config.buffer_type)
    out_layout = ttnn.TILE_LAYOUT if out_tiled else ttnn.ROW_MAJOR_LAYOUT
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape(indices_shape, table_shape)), weight.dtype, out_layout, input_tensor.device(), out_config
    )
    spec, run_args, tensor_indices = create_program_artifacts(input_tensor, weight, output, **plan_overrides)
    return ttnn.generic_op([input_tensor, weight, output], spec, run_args, tensor_indices)
