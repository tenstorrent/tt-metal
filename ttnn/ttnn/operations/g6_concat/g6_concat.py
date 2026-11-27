# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_concat: torch.cat(tensors, dim) as one device program (reader + writer, no compute)."""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_concat_program_descriptor as pd


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — each reads the scenario dict
#    {"shapes", "dim", "dtype", "layout", "input_buffers", "memory_config"}
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs["dtype"]


def tag_layout(inputs, axes):
    return inputs["layout"]


def tag_concat_dim(inputs, axes):
    rank = len(inputs["shapes"][0])
    d = pd.canonical_dim(inputs["dim"], rank)
    if d == rank - 1:
        return "w"
    if d == rank - 2:
        return "h"
    return "outer"


def tag_alignment(inputs, axes):
    shapes = inputs["shapes"]
    if any(s[-1] % 32 != 0 for s in shapes):
        return "w_non_aligned"
    if any(s[-2] % 32 != 0 for s in shapes):
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


def _scenario(tensors, dim, memory_config):
    out_l1 = memory_config is not None and memory_config.buffer_type == ttnn.BufferType.L1
    return {
        "shapes": [list(t.shape) for t in tensors],
        "dim": dim,
        "dtype": tensors[0].dtype,
        "layout": tensors[0].layout,
        "input_buffers": [_buffer_name(t.memory_config().buffer_type) for t in tensors],
        "memory_config": "l1" if out_l1 else None,
    }


def _check_structure(tensors, dim, memory_config):
    if not isinstance(tensors, (list, tuple)) or len(tensors) == 0:
        raise ValueError("g6_concat: tensors must be a non-empty list")
    if len(tensors) > pd.K_MAX:
        raise RuntimeError(f"g6_concat: {len(tensors)} inputs exceed K_MAX={pd.K_MAX} (runtime-arg capacity)")
    shapes = [list(t.shape) for t in tensors]
    rank = len(shapes[0])
    if rank not in (3, 4):
        raise ValueError(f"g6_concat: rank {rank} not in (3, 4)")
    if any(len(s) != rank for s in shapes):
        raise ValueError(f"g6_concat: inputs differ in rank: {shapes}")
    if any(t.dtype != tensors[0].dtype for t in tensors):
        raise ValueError("g6_concat: inputs differ in dtype")
    if any(t.layout != tensors[0].layout for t in tensors):
        raise ValueError("g6_concat: inputs differ in layout")
    if not isinstance(dim, int) or not -rank <= dim <= rank - 1:
        raise ValueError(f"g6_concat: dim={dim!r} outside [{-rank}, {rank - 1}]")
    d = pd.canonical_dim(dim, rank)
    for s in shapes[1:]:
        if any(s[i] != shapes[0][i] for i in range(rank) if i != d):
            raise ValueError(f"g6_concat: sizes differ outside dim {d}: {shapes}")
    for t in tensors:
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise ValueError("g6_concat: sharded inputs are not accepted; every input must be interleaved")
    if memory_config is not None and memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("g6_concat: memory_config must be INTERLEAVED")
    return d


def validate(tensors, dim=0, *, memory_config=None):
    _check_structure(tensors, dim, memory_config)

    scenario = _scenario(tensors, dim, memory_config)
    axes = {}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(scenario, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_concat: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_concat: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Program cache (BB-07): key -> {plan, kernels, descriptor}
# ---------------------------------------------------------------------------

_PROGRAM_CACHE = {}


def _output_shape(tensors, d):
    shape = list(tensors[0].shape)
    shape[d] = sum(int(t.shape[d]) for t in tensors)
    return shape


def _cache_key(tensors, d, output_memory_config, device):
    grid = device.compute_with_storage_grid_size()
    return (
        tuple(tuple(t.shape) for t in tensors),
        d,
        tensors[0].dtype,
        tensors[0].layout,
        tuple(t.memory_config().buffer_type for t in tensors),
        output_memory_config.buffer_type,
        output_memory_config.memory_layout,
        (grid.x, grid.y),
        pd.MAX_CORES,
        pd.BLOCK_BYTES_CAP,
        pd.OUTPUT_PAGES_DEPTH,
        pd.TRIM_ROWS_MAX,
    )


def _entry(tensors, d, output, output_memory_config, device):
    key = _cache_key(tensors, d, output_memory_config, device)
    entry = _PROGRAM_CACHE.get(key)
    if entry is None:
        plan = pd.build_plan(tensors, d, _output_shape(tensors, d), output_memory_config, device)
        if plan["l1_total"] > pd.L1_OP_BUDGET:
            raise RuntimeError(
                f"g6_concat: L1 use {plan['l1_total']} B per core exceeds the op budget {pd.L1_OP_BUDGET} B"
            )
        if output is None:
            return plan, None
        entry = pd.build_descriptor(plan, tensors, output, hash(key) & 0xFFFFFFFFFFFFFFFF)
        _PROGRAM_CACHE[key] = entry
        return plan, entry["descriptor"]
    if output is None:
        return entry["plan"], None
    return entry["plan"], pd.refresh_addresses(entry, tensors, output)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_concat(tensors, dim=0, *, memory_config=None):
    validate(tensors, dim, memory_config=memory_config)

    tensors = list(tensors)
    d = pd.canonical_dim(dim, len(tensors[0].shape))
    device = tensors[0].device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(_output_shape(tensors, d)),
        tensors[0].dtype,
        tensors[0].layout,
        device,
        output_memory_config,
    )
    _, descriptor = _entry(tensors, d, output, output_memory_config, device)
    return ttnn.generic_op([*tensors, output], descriptor)


def g6_concat_plan(tensors, dim=0, *, memory_config=None):
    """The plan a call selects (BB-08 / BB-21); dispatches nothing."""
    validate(tensors, dim, memory_config=memory_config)
    tensors = list(tensors)
    d = pd.canonical_dim(dim, len(tensors[0].shape))
    device = tensors[0].device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    plan, _ = _entry(tensors, d, None, output_memory_config, device)
    role = f"{plan['regime']} reader+writer"
    return {
        "regime": plan["regime"],
        "num_cores": plan["num_cores"],
        "core_ranges": str(plan["all_cores"]),
        "block_pages": plan["block_pages"],
        "depth": plan["depth"],
        "cb_output_bytes": plan["cb_output_bytes"],
        "cb_staging_bytes": plan["cb_staging_bytes"],
        "l1_total": plan["l1_total"],
        "roles": [
            {"core": (c.x, c.y), "role": role, "pages": (plan["starts"][i], plan["starts"][i] + plan["counts"][i])}
            for i, c in enumerate(plan["cores"])
        ],
    }
