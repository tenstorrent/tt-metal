# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""nlp_concat_heads: put the heads of an attention output side by side.

    input  [B, H, S, D]  ->  output [B, 1, S, H * D],  output[b, 0, s, h * D + d] = input[b, h, s, d]

One `ttnn.generic_op` dispatch of one `ttnn.ProgramSpec` (nlp_concat_heads_program_artifacts.py). The
specification is eval/specs/nlp_concat_heads/spec.py.
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval import memory
from eval.specs.nlp_concat_heads.spec import AXES, canonical_memory

from .nlp_concat_heads_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase (the specification's PHASE0)
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":         [ttnn.bfloat16],
    "layout":        [ttnn.TILE_LAYOUT],
    "memory_config": ["DRAM"],
    "sharding":      ["interleaved"],
    "output_memory": ["DRAM"],
    "alignment":     ["tile_aligned"],
    "batch":         ["single"],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _axes(input_tensor, memory_config):
    shape = tuple(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"nlp_concat_heads: the input must be [B, num_heads, S, head_dim] (rank 4), got {list(shape)}")
    placed = memory.memory_axes(input_tensor)
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "memory_config": placed["memory_config"],
        "sharding": placed["sharding"],
        # TypeError for anything that is not a memory config; "<buffer>/<sharding>" for a sharded one.
        "output_memory": canonical_memory(memory_config, {}),
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)
    return axes


def validate(input_tensor, *, memory_config=None):
    axes = _axes(input_tensor, memory_config)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"nlp_concat_heads: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"nlp_concat_heads: unsupported combination (refinement candidate): {exc}")
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _output_memory_config(input_tensor, memory_config):
    """Rule R3: where `memory_config` says, interleaved; None: the input's buffer, interleaved."""
    if memory_config is not None:
        return memory_config
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, input_tensor.memory_config().buffer_type)


def nlp_concat_heads(
    input_tensor: ttnn.Tensor,
    *,
    memory_config: "ttnn.MemoryConfig" = None,
) -> ttnn.Tensor:
    validate(input_tensor, memory_config=memory_config)

    batch, heads, tokens, head_dim = (int(d) for d in input_tensor.shape)
    out_spec = ttnn.TensorSpec(
        ttnn.Shape([batch, 1, tokens, heads * head_dim]),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        memory_config=_output_memory_config(input_tensor, memory_config),
    )
    output_tensor = ttnn.allocate_tensor_on_device(out_spec, input_tensor.device())
    spec, run_args, tensor_indices = create_program_artifacts(input_tensor, output_tensor)
    return ttnn.generic_op([input_tensor, output_tensor], spec, run_args, tensor_indices)
