# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""to_memory_config: the input's values, moved to the memory config the caller names.

Specification: eval/specs/to_memory_config/spec.py. Design: op_design.md beside this file.

The op moves pages and computes nothing (rule R1): page `p` of the output is page `p` of the input, read
through the input's TensorAccessor and written through the output's. The same reader/writer pair serves
interleaved <-> interleaved, interleaved <-> height- or width-sharded and one shard grid to another (rule R6);
only the table of (core, first entry, entry count, run) changes, see to_memory_config_program_artifacts.py. In
row-major layout a width shard's page is a run of one row, so the ring carries row runs a side may assemble from
or split into several pages (the module docstring there).

Exactly one device dispatch per call (`ttnn.generic_op`), also when the input already has the asked-for
config (rule R7: a copy is returned).
"""

from __future__ import annotations

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval import memory
from eval.specs.to_memory_config.spec import AXES, canonical_to

from .to_memory_config_program_artifacts import NOC_MAX_PACKET_BYTES, create_program_artifacts

OP_NAME = "to_memory_config"

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the specification's PHASE0, plus width sharding on both sides (Refinement 0).
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":         [ttnn.bfloat16],
    "layout":        [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "memory_config": ["DRAM", "L1"],
    "sharding":      ["interleaved", "height", "width"],
    "to":            ["DRAM", "L1", "L1/height", "L1/width"],
    "alignment":     ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank":          [2, 3, 4],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: none in the first phase.
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _axes(tensor, memory_config) -> dict:
    """The cell of a call, derived as the golden runner derives it."""
    # `canonical_to` raises TypeError for anything that is not a memory config (X1: "L1", X2: None).
    to = canonical_to(memory_config, {"tensor": tensor})
    axes = {
        "dtype": tensor.dtype,
        "layout": tensor.layout,
        **memory.memory_axes(tensor),   # "memory_config", "sharding" of the input (rule R5)
        "to": to,
    }
    shape = tuple(int(d) for d in tensor.shape)
    if len(shape) < 2:
        # The taggers index H and W; a rank-1 tensor is outside the rank axis before they run.
        raise UnsupportedAxisValue(f"{OP_NAME}: rank={len(shape)} not in SUPPORTED {SUPPORTED['rank']}")
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)
    return axes


def validate(tensor, memory_config) -> dict:
    """Refuse anything outside the first phase (rule R8); returns the call's axes."""
    axes = _axes(tensor, memory_config)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{OP_NAME}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"{OP_NAME}: unsupported combination (refinement candidate): {exc}")

    # Mechanism caps (op_design.md, Mechanism caps). Not axis values: limits of the one data path.
    if axes["sharding"] != "interleaved" and axes["memory_config"] != "L1":
        # A DRAM-sharded tensor's shard grid is DRAM banks, not worker cores (the spec's INVALID cell).
        raise NotImplementedError(f"{OP_NAME}: a {axes['memory_config']}/{axes['sharding']} input is not built")
    if tensor.buffer_aligned_page_size() > NOC_MAX_PACKET_BYTES:
        # One implicit-sync transfer per page is one NoC packet: a page above the packet cap is not served.
        raise NotImplementedError(
            f"{OP_NAME}: a page of {tensor.buffer_aligned_page_size()} B exceeds the {NOC_MAX_PACKET_BYTES} B "
            "of one NoC packet")
    return axes


# ---------------------------------------------------------------------------
# Public entry point: the specification's stub.
# ---------------------------------------------------------------------------


def to_memory_config(tensor: ttnn.Tensor, memory_config: "ttnn.MemoryConfig") -> ttnn.Tensor:
    validate(tensor, memory_config)
    device = tensor.device()
    out_spec = ttnn.TensorSpec(tensor.shape, tensor.dtype, tensor.layout, memory_config, tile=tensor.tile)
    output = ttnn.allocate_tensor_on_device(out_spec, device)
    spec, run_args, tensor_indices = create_program_artifacts(tensor, output)
    return ttnn.generic_op([tensor, output], spec, run_args, tensor_indices)
