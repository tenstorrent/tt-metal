# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_concat: torch.cat(tensors, dim) on device tensors, in one device program.

Every logical input element is read once from its own buffer and every output element is written
once, bit for bit. Design: op_design.md (Blocking Model) next to this file.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_concat_program_descriptor as pdesc

TILE = 32


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — `inputs` is one scenario dict:
#    {"shapes", "dim", "dtype", "layout", "input_buffers", "memory_config"}
# ---------------------------------------------------------------------------


def _canonical_dim(inputs):
    rank = len(inputs["shapes"][0])
    return pdesc.canonical_dim(inputs["dim"], rank)


def tag_dtype(inputs, axes):
    return inputs["dtype"]


def tag_layout(inputs, axes):
    return inputs["layout"]


def tag_concat_dim(inputs, axes):
    rank = len(inputs["shapes"][0])
    d = _canonical_dim(inputs)
    if d == rank - 1:
        return "w"
    if d == rank - 2:
        return "h"
    return "outer"


def tag_alignment(inputs, axes):
    shapes = inputs["shapes"]
    if any(s[-1] % TILE != 0 for s in shapes):
        return "w_non_aligned"
    if any(s[-2] % TILE != 0 for s in shapes):
        return "h_non_aligned"
    return "tile_aligned"


def tag_input_buffer(inputs, axes):
    buffers = set(inputs["input_buffers"])
    if buffers == {"dram"}:
        return "dram"
    if buffers == {"l1"}:
        return "l1"
    return "mixed"


def tag_output_buffer(inputs, axes):
    return "l1" if inputs["memory_config"] == "l1" else "dram"


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "concat_dim": tag_concat_dim,
    "alignment": tag_alignment,
    "input_buffer": tag_input_buffer,
    "output_buffer": tag_output_buffer,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "concat_dim": ["outer", "h", "w"],
    "alignment": ["tile_aligned", "h_non_aligned"],
    "input_buffer": ["dram"],
    "output_buffer": ["dram"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _buffer_name(buffer_type):
    return "l1" if buffer_type == ttnn.BufferType.L1 else "dram"


def _scenario_of_call(tensors, dim, memory_config):
    """The spec's scenario dict, built exactly as eval/golden_tests/g6_concat/axes.py builds it."""
    out_l1 = memory_config is not None and memory_config.buffer_type == ttnn.BufferType.L1
    return {
        "shapes": [list(t.shape) for t in tensors],
        "dim": dim,
        "dtype": tensors[0].dtype,
        "layout": tensors[0].layout,
        "input_buffers": [_buffer_name(t.memory_config().buffer_type) for t in tensors],
        "memory_config": "l1" if out_l1 else None,
    }


def _validate_structure(tensors, dim, memory_config):
    if not isinstance(tensors, (list, tuple)) or len(tensors) == 0:
        raise ValueError("g6_concat: `tensors` must be a non-empty list of device tensors")
    tensors = list(tensors)
    for t in tensors:
        if not isinstance(t, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(t):
            raise ValueError("g6_concat: every input must be a ttnn.Tensor on device")
    if len(tensors) > pdesc.K_MAX:
        raise RuntimeError(f"g6_concat: {len(tensors)} inputs exceed K_MAX = {pdesc.K_MAX} (runtime-arg capacity)")

    shapes = [list(t.shape) for t in tensors]
    rank = len(shapes[0])
    if rank not in (3, 4):
        raise ValueError(f"g6_concat: rank must be 3 or 4, got {rank}")
    if any(len(s) != rank for s in shapes):
        raise ValueError(f"g6_concat: all inputs must have the same rank, got {[len(s) for s in shapes]}")
    if any(t.dtype != tensors[0].dtype for t in tensors):
        raise ValueError(f"g6_concat: all inputs must share one dtype, got {[t.dtype for t in tensors]}")
    if any(t.layout != tensors[0].layout for t in tensors):
        raise ValueError(f"g6_concat: all inputs must share one layout, got {[t.layout for t in tensors]}")
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise ValueError(f"g6_concat: dim must be an int, got {dim!r}")
    if not -rank <= dim <= rank - 1:
        raise ValueError(f"g6_concat: dim {dim} outside [{-rank}, {rank - 1}] for rank {rank}")
    d = pdesc.canonical_dim(dim, rank)
    for s in shapes:
        if any(x == 0 for x in s):
            raise ValueError(f"g6_concat: empty inputs are not accepted, got shape {s}")
        if [x for i, x in enumerate(s) if i != d] != [x for i, x in enumerate(shapes[0]) if i != d]:
            raise ValueError(f"g6_concat: input sizes differ outside dim {d}: {shapes}")
    device_id = tensors[0].device().id()
    if any(t.device().id() != device_id for t in tensors):
        raise ValueError("g6_concat: all inputs must live on the same device")
    for t in tensors:
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise ValueError("g6_concat: sharded inputs are not accepted; every input must be interleaved")
    if memory_config is not None and memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("g6_concat: memory_config must be INTERLEAVED")
    return d


def validate(tensors, dim=0, *, memory_config=None):
    """Structural refusals (ValueError / RuntimeError), then the SUPPORTED / EXCLUSIONS gate.

    Returns the canonical dim.
    """
    d = _validate_structure(tensors, dim, memory_config)
    tensors = list(tensors)

    scenario = _scenario_of_call(tensors, dim, memory_config)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(scenario, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_concat: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_concat: unsupported combination (refinement candidate): {exc}")

    if axes["layout"] == ttnn.TILE_LAYOUT and tuple(tensors[0].tile.tile_shape) != (TILE, TILE):
        raise UnsupportedAxisValue(f"g6_concat: only 32x32 tiles are supported, got {tensors[0].tile.tile_shape}")
    return d


# ---------------------------------------------------------------------------
# Program cache (BB-07): one descriptor per key, built once; a hit rewrites only the
# buffer-address common runtime args.
# ---------------------------------------------------------------------------


class _CacheEntry:
    def __init__(self, plan):
        self.plan = plan
        self.descriptor = None
        self.reader = None
        self.writer = None


_PROGRAM_CACHE = {}


def _output_memory_config(memory_config):
    return memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG


def _cache_key(tensors, d, out_mc):
    device = tensors[0].device()
    grid = device.compute_with_storage_grid_size()
    return (
        device.id(),
        tuple(tuple(int(x) for x in t.shape) for t in tensors),
        d,
        str(tensors[0].dtype),
        str(tensors[0].layout),
        tuple(str(t.memory_config().buffer_type) for t in tensors),
        str(out_mc.buffer_type),
        str(out_mc.memory_layout),
        (grid.x, grid.y),
        pdesc.MAX_CORES,
        pdesc.BLOCK_BYTES_CAP,
        pdesc.OUTPUT_PAGES_DEPTH,
    )


def _cache_entry(tensors, d, out_mc):
    key = _cache_key(tensors, d, out_mc)
    entry = _PROGRAM_CACHE.get(key)
    if entry is None:
        entry = _CacheEntry(pdesc.make_plan(tensors, d, out_mc))
        _PROGRAM_CACHE[key] = entry
    return key, entry


def g6_concat_plan(tensors, dim=0, *, memory_config=None):
    """The plan a call with these arguments selects (BB-08/BB-21). Dispatches nothing."""
    d = validate(tensors, dim, memory_config=memory_config)
    _, entry = _cache_entry(list(tensors), d, _output_memory_config(memory_config))
    return entry.plan.report()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_concat(tensors, dim=0, *, memory_config=None):
    """Concatenate `tensors` along `dim` (torch.cat semantics) in one device program.

    Args:
        tensors: K >= 1 device tensors of rank 3 or 4, one dtype and layout, equal sizes outside dim.
        dim: axis to join along; negative counts from the end.
        memory_config: placement of the output; None gives DRAM interleaved.

    Returns:
        The output tensor: shape of tensors[0] with the summed size in dim.
    """
    d = validate(tensors, dim, memory_config=memory_config)
    tensors = list(tensors)
    out_mc = _output_memory_config(memory_config)
    key, entry = _cache_entry(tensors, d, out_mc)
    plan = entry.plan

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(plan.out_shape), tensors[0].dtype, tensors[0].layout, tensors[0].device(), out_mc
    )

    if entry.descriptor is None:
        entry.descriptor, entry.reader, entry.writer = pdesc.create_program_descriptor(plan, tensors, output, key)
    else:
        reader_common = list(entry.reader.common_runtime_args)
        for t, index in zip(tensors, pdesc.reader_address_indices(len(tensors))):
            reader_common[index] = t.buffer_address()
        entry.reader.common_runtime_args = reader_common
        writer_common = list(entry.writer.common_runtime_args)
        writer_common[pdesc.WRITER_ADDR_INDEX] = output.buffer_address()
        entry.writer.common_runtime_args = writer_common
        entry.descriptor.kernels = [entry.reader, entry.writer]

    return ttnn.generic_op([*tensors, output], entry.descriptor)
