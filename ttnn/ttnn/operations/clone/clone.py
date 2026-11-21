# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""clone: a copy of a tensor, placed where the caller says (eval/specs/clone/spec.py).

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate(), then the
entry point. The program is one Metal 2.0 ProgramSpec (clone_program_artifacts.py) dispatched once
with ttnn.generic_op: a reader and a writer data-movement kernel per cluster, joined by one ring.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, SupportRefusal, UnsupportedAxisValue

from eval import memory
from eval.specs.clone.spec import AXES, canonical_dtype, canonical_memory

from .clone_program_artifacts import create_program_artifacts, entry_bytes_of


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES


# ---------------------------------------------------------------------------
# 2. SUPPORTED: the specification's PHASE0, axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":          [ttnn.bfloat16],
    "layout":         [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "memory_config":  ["DRAM", "L1"],
    "sharding":       ["interleaved"],
    "dtype_argument": [None, "same"],
    "output_memory":  ["L1"],
    "alignment":      ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank":           [2, 3, 4],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

# One implicit-sync ring entry is one NoC transaction, and the DM0 interrupt handler counts
# transactions: a page above one NoC packet would be credited on its first fragment.
# NOC_MAX_BURST_SIZE = 256 words x 256 B (tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:395-396;
# tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:560). There is no Python query for it.
NOC_MAX_BURST_BYTES = 256 * 256


def _axes(input, dtype, memory_config):
    """The cell of a call, built as the golden suite builds it. Canonicalization comes first so a
    `memory_config` that is not a config object (X1) raises TypeError before anything else."""
    output_memory = canonical_memory(memory_config, {"input": input})
    dtype_argument = canonical_dtype(dtype, {"input": input})
    placed = memory.memory_axes(input)
    axes = {
        "dtype": input.dtype,
        "layout": input.layout,
        "memory_config": placed["memory_config"],
        "sharding": placed["sharding"],
        "dtype_argument": dtype_argument,
        "output_memory": output_memory,
    }
    shape = tuple(input.shape)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)
    return axes


def validate(input, *, dtype=None, memory_config=None):
    if not isinstance(input, ttnn.Tensor):
        raise TypeError(f"clone: input must be a ttnn.Tensor, got {type(input).__name__}")
    axes = _axes(input, dtype, memory_config)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"clone: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"clone: unsupported combination (refinement candidate): {exc}")

    # Mechanism cap (op_design.md, Mechanism caps): the `oversized_page` regime is not built. A typed
    # support refusal (a NotImplementedError), so the harness reads it as deliberate, like the two above.
    entry_bytes = entry_bytes_of(input)
    if entry_bytes > NOC_MAX_BURST_BYTES:
        raise SupportRefusal(
            f"clone: a page of {entry_bytes} B exceeds one NoC packet ({NOC_MAX_BURST_BYTES} B); "
            "the oversized-page regime is not built"
        )
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _output_memory_config(input, memory_config):
    """Where the copy goes (R3): the config asked for, else the input's buffer, interleaved."""
    if memory_config is not None:
        return memory_config
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, input.memory_config().buffer_type)


def clone(
    input: ttnn.Tensor,
    *,
    dtype: ttnn.DataType = None,
    memory_config: "ttnn.MemoryConfig" = None,
) -> ttnn.Tensor:
    validate(input, dtype=dtype, memory_config=memory_config)

    # Always a fresh buffer, also when the destination is the input's own buffer (R5).
    out_spec = ttnn.TensorSpec(
        input.shape, input.dtype, input.layout, _output_memory_config(input, memory_config), tile=input.tile
    )
    output = ttnn.allocate_tensor_on_device(out_spec, input.device())

    spec, run_args, tensor_indices = create_program_artifacts(input, output)
    return ttnn.generic_op([input, output], spec, run_args, tensor_indices)
