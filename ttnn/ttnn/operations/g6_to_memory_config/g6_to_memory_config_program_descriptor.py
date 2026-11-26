# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for g6_to_memory_config.

Three regimes of one scheme — the mover of a page is a core that already holds one end of it
(op_design.md, Blocking Model):

  PULL  (target L1-sharded)   one kernel on each output-shard core, NCRISC/NoC0; block = its output shard
  PUSH  (source L1-sharded)   one kernel on each input-shard core,  BRISC/NoC1;  block = its input shard
  STAGE (both ends are banks) reader NCRISC/NoC0 + writer BRISC/NoC1 on a full-grid split;
                              block = STAGE_BLOCK_PAGES_CAP-capped flat page range, ring depth STAGE_DEPTH

Knobs (single source each): STAGE_BLOCK_PAGES_CAP, STAGE_DEPTH, SHARD_BLOCKS_PER_CORE (host constants,
threaded as CT/RT args); the STAGE grid is device.compute_with_storage_grid_size() through
split_work_to_cores. Per-core shard rectangles are RT args.

BB-07 cache: the address-independent plan (core ranges, CT args, per-core RT args with an address
slot) is memoized per (regime, shape, dtype, both memory configs, device grid); only the buffer
addresses are patched per call.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# ---- knobs (single source of truth) ----
STAGE_BLOCK_PAGES_CAP = 16  # max pages per STAGE block (one read barrier / CB handshake per block)
STAGE_DEPTH = 2  # STAGE ring depth in blocks (reader fills block k+1 while writer drains block k)
SHARD_BLOCKS_PER_CORE = 1  # PULL/PUSH: blocks per shard owner (the whole resident shard is one block)

# BB-05 per-core L1 budget for op buffers + output L1 pages.
L1_BUDGET_BYTES = 1347776

# ---- CB slots ----
CB_STAGED_PAGES = 0
CB_OUTPUT_SHARD = 1
CB_INPUT_SHARD = 2

_PLAN_CACHE: dict = {}


def select_regime(input_tensor, memory_config) -> str:
    """The plan a call takes (public, BB-08). Same predicate the dispatch uses."""
    if memory_config.is_sharded() and memory_config.buffer_type == ttnn.BufferType.L1:
        return "pull"
    in_config = input_tensor.memory_config()
    if in_config.is_sharded() and in_config.buffer_type == ttnn.BufferType.L1:
        return "push"
    return "stage"


def _tile_grid(shape):
    leading = 1
    for d in shape[:-2]:
        leading *= d
    return leading * ((shape[-2] + TILE - 1) // TILE), (shape[-1] + TILE - 1) // TILE


def _shard_key(memory_config):
    spec = memory_config.shard_spec if memory_config.is_sharded() else None
    if spec is None:
        return (memory_config.buffer_type, memory_config.memory_layout, None)
    ranges = tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in spec.grid.ranges())
    return (memory_config.buffer_type, memory_config.memory_layout, ranges, tuple(spec.shape), spec.orientation)


def _merged(grid):
    """Shard grids are often written as many single-core ranges; merge into rectangles (BB-17)."""
    try:
        return grid.merge(grid)
    except Exception:  # noqa: BLE001 — merging is an optimization; the raw grid is equivalent
        return grid


def _shard_blocks(tensor, tensor_tile_rows, tensor_tile_cols):
    """Per-core shard rectangles of a sharded tensor.

    Returns (grid, shard_tile_cols_nominal, [(core, num_blocks, row0, rows, col0, cols), ...]).
    height/width: the i-th core of corerange_to_cores(grid, row_wise=True) holds shard i (row-major
    over the shard grid). block (ROW_MAJOR): core at bbox offset (dx, dy) holds shard (dy, dx).
    Extents are the actual ones (ragged last shard); cores past the last shard get 0 blocks.
    """
    memory_config = tensor.memory_config()
    spec = memory_config.shard_spec
    grid = spec.grid
    shard_tile_rows = spec.shape[0] // TILE
    shard_tile_cols = spec.shape[1] // TILE
    n_shard_rows = (tensor_tile_rows + shard_tile_rows - 1) // shard_tile_rows
    n_shard_cols = (tensor_tile_cols + shard_tile_cols - 1) // shard_tile_cols
    cores = list(ttnn.corerange_to_cores(grid, None, True))

    blocks = []
    if memory_config.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        bbox = grid.bounding_box()
        x0, y0 = bbox.start.x, bbox.start.y
        shard_of = [(c.y - y0, c.x - x0) for c in cores]
    else:
        shard_of = [(i // n_shard_cols, i % n_shard_cols) for i in range(len(cores))]

    for core, (sr, sc) in zip(cores, shard_of):
        row0 = sr * shard_tile_rows
        col0 = sc * shard_tile_cols
        if sr < n_shard_rows and sc < n_shard_cols:
            rows = min(shard_tile_rows, tensor_tile_rows - row0)
            cols = min(shard_tile_cols, tensor_tile_cols - col0)
            blocks.append((core, SHARD_BLOCKS_PER_CORE, row0, rows, col0, cols))
        else:
            blocks.append((core, 0, 0, 0, 0, 0))
    return grid, shard_tile_cols, blocks


def _output_l1_bytes_per_core(output_tensor, page_size, num_pages):
    config = output_tensor.memory_config()
    if config.buffer_type != ttnn.BufferType.L1:
        return 0
    if config.is_sharded():
        sh, sw = config.shard_spec.shape
        return (sh // TILE) * (sw // TILE) * page_size
    num_l1_banks = output_tensor.device().compute_with_storage_grid_size()
    banks = max(1, num_l1_banks.x * num_l1_banks.y)
    return ((num_pages + banks - 1) // banks) * page_size


def _build_plan(regime, input_tensor, output_tensor):
    page_size = input_tensor.buffer_page_size()
    num_pages = input_tensor.buffer_num_pages()
    if output_tensor.buffer_page_size() != page_size or output_tensor.buffer_num_pages() != num_pages:
        raise ValueError("g6_to_memory_config: input and output page geometry differ")
    if input_tensor.buffer_aligned_page_size() != page_size or output_tensor.buffer_aligned_page_size() != page_size:
        raise ValueError("g6_to_memory_config: tile page size must equal its aligned page size")
    tensor_tile_rows, tensor_tile_cols = _tile_grid(list(input_tensor.shape))
    assert tensor_tile_rows * tensor_tile_cols == num_pages, (tensor_tile_rows, tensor_tile_cols, num_pages)

    output_l1 = _output_l1_bytes_per_core(output_tensor, page_size, num_pages)

    if regime in ("pull", "push"):
        if regime == "pull":
            shard_tensor, other_tensor, cb_index = output_tensor, input_tensor, CB_OUTPUT_SHARD
            source, config = "g6_to_memory_config_pull.cpp", ttnn.ReaderConfigDescriptor()
        else:
            shard_tensor, other_tensor, cb_index = input_tensor, output_tensor, CB_INPUT_SHARD
            source, config = "g6_to_memory_config_push.cpp", ttnn.WriterConfigDescriptor()
        if output_l1 > L1_BUDGET_BYTES:
            raise ValueError(f"g6_to_memory_config: output needs {output_l1} B of L1 per core > {L1_BUDGET_BYTES}")
        grid, shard_tile_cols, blocks = _shard_blocks(shard_tensor, tensor_tile_rows, tensor_tile_cols)
        ct_args = [page_size, tensor_tile_cols, shard_tile_cols, cb_index]
        ct_args.extend(ttnn.TensorAccessorArgs(other_tensor).get_compile_time_args())
        per_core = [(core, list(rest)) for core, *rest in blocks]
        return {
            "regime": regime,
            "core_ranges": _merged(grid),
            "kernels": [(source, config, ct_args, per_core, "other")],
            "cb": ("sharded", cb_index, "shard"),
        }

    # STAGE
    grid_size = input_tensor.device().compute_with_storage_grid_size()
    (_, all_cores, group_1, group_2, pages_1, pages_2) = ttnn.split_work_to_cores(grid_size, num_pages, row_wise=True)
    core_pages_max = max(pages_1, pages_2)
    stage_block_pages = max(1, min(core_pages_max, STAGE_BLOCK_PAGES_CAP))
    op_l1 = STAGE_DEPTH * stage_block_pages * page_size
    if op_l1 + output_l1 > L1_BUDGET_BYTES:
        raise ValueError(f"g6_to_memory_config: {op_l1} + {output_l1} B of L1 per core > {L1_BUDGET_BYTES}")
    per_core = []
    start = 0
    for group, count in ((group_1, pages_1), (group_2, pages_2)):
        if count == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            per_core.append((core, [start, count]))
            start += count
    assert start == num_pages, (start, num_pages)
    reader_ct = [page_size, stage_block_pages, CB_STAGED_PAGES]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct = [page_size, stage_block_pages, CB_STAGED_PAGES]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    return {
        "regime": regime,
        "core_ranges": all_cores,
        "kernels": [
            ("g6_to_memory_config_reader.cpp", ttnn.ReaderConfigDescriptor(), reader_ct, per_core, "input"),
            ("g6_to_memory_config_writer.cpp", ttnn.WriterConfigDescriptor(), writer_ct, per_core, "output"),
        ],
        "cb": ("ring", CB_STAGED_PAGES, STAGE_DEPTH * stage_block_pages, page_size),
    }


def create_program_descriptor(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> ttnn.ProgramDescriptor:
    out_config = output_tensor.memory_config()
    regime = select_regime(input_tensor, out_config)
    grid_size = input_tensor.device().compute_with_storage_grid_size()
    key = (
        regime,
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        _shard_key(input_tensor.memory_config()),
        _shard_key(out_config),
        (grid_size.x, grid_size.y),
    )
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _build_plan(regime, input_tensor, output_tensor)
        _PLAN_CACHE[key] = plan

    core_ranges = plan["core_ranges"]
    addresses = {"input": input_tensor.buffer_address(), "output": output_tensor.buffer_address()}
    addresses["other"] = addresses["output"] if regime == "push" else addresses["input"]

    kernels = []
    for source, config, ct_args, per_core, addr_of in plan["kernels"]:
        rt_args = ttnn.RuntimeArgs()
        address = addresses[addr_of]
        for core, args in per_core:
            rt_args[core.x][core.y] = [address, *args]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / source),
                core_ranges=core_ranges,
                compile_time_args=ct_args,
                runtime_args=rt_args,
                config=config,
            )
        )

    cb_plan = plan["cb"]
    if cb_plan[0] == "sharded":
        shard_tensor = output_tensor if regime == "pull" else input_tensor
        cbs = [ttnn.cb_descriptor_from_sharded_tensor(cb_plan[1], shard_tensor, core_ranges=core_ranges)]
    else:
        _, cb_index, num_pages, page_size = cb_plan
        cbs = [
            ttnn.CBDescriptor(
                total_size=num_pages * page_size,
                core_ranges=core_ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=cb_index, data_format=input_tensor.dtype, page_size=page_size)
                ],
            )
        ]

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
