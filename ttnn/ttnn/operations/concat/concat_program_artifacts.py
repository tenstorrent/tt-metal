# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for concat (op_design.md: Blocking Model, Work Distribution).

One reader and one writer per cluster, no compute kernel (the op moves bytes only), one ring between them.

    out_tile (flat output page axis)  split over the grid in contiguous ranges (split_work_to_cores)
    block                             BLOCK_TILES output pages per reader reservation (run_assembly)
    ring cb_output_tiles              OUTPUT_DEPTH blocks resident: the reader fills one while the writer drains one

Four regimes, chosen from the layout and the shapes alone (select_regime). TILE layout (page = tile):

    page_copy       every seam on a tile boundary: each output tile is one whole input page (implicit-sync reader)
    run_assembly    some seam inside a tile: each output tile is assembled from sub-page byte runs read straight
                    into the ring entry (explicit-sync reader: reserve, tagged reads, one barrier, push per block)

ROW_MAJOR layout (page = stick, one row of the last dimension; op_design.md's row_major_runs regime):

    row_major_copy  dim=-2: each output stick is one whole input stick (page_copy in the stick domain)
    row_major_runs  dim=-1: each output stick is one byte run per piece, at the piece's element offset (run_assembly
                    in the stick domain: explicit sync, one barrier per block of sticks)

The ring entry is one output page: a tile, or a stick rounded up to the L1 alignment (never more than any input's
or the output's aligned page, so every whole-entry transfer stays inside its page).
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

TILE = 32
# The cluster model of the target page; Python has no query for it.
RING_MAX_BYTES = 1_048_560  # one DFB ring at most 1 MiB - 16 B
MAX_INPUTS = 5  # TARGET's largest count: the reader binds in_0 .. in_{count-1}
L1_RESERVE_BYTES = 256 * 1024  # left free beside the ring (L1 input tensors live in the same L1)

# Blocking knobs (op_design.md: Blocking Model). Each is defined once; everything below derives from them.
# BLOCK_TILES = 8 output tiles per reader reservation (run_assembly). In ROW_MAJOR
# the block is counted in sticks and holds the same bytes as BLOCK_TILES tiles (default_block_pages), so a narrow
# stick does not pay a barrier per few hundred bytes and a wide one is clamped like a tile block.
BLOCK_TILES = 8
OUTPUT_DEPTH = 2  # blocks resident in cb_output_tiles: double buffering between reader and writer

# Thread plan (R, C, W) = (1, 0, 1): bound by data movement, so one reader and one writer and no compute kernel.
THREAD_PLAN = {"reader_threads": 1, "writer_threads": 1}

REGIMES = {"page_copy": 0, "run_assembly": 1, "row_major_copy": 2, "row_major_runs": 3}
EXPLICIT_REGIMES = ("run_assembly", "row_major_runs")  # an entry assembled from several transfers
RING = "cb_output_tiles"


def select_regime(shapes, dim, layout=ttnn.TILE_LAYOUT):
    """TILE: page_copy when every seam is on a tile boundary (every piece but the last a multiple of 32 along dim),
    run_assembly otherwise. ROW_MAJOR: row_major_copy along H (whole sticks), row_major_runs along W."""
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        return "row_major_copy" if dim == -2 else "row_major_runs"
    return "page_copy" if all(s[dim] % TILE == 0 for s in shapes[:-1]) else "run_assembly"


def page_geometry(out, layout):
    """(entry bytes, element bytes) of one output page: a tile, or a stick rounded up to the L1 alignment."""
    elem_bytes = out.element_size()
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        align = ttnn.get_l1_alignment()
        stick = int(out.shape[-1]) * elem_bytes
        entry = -(-stick // align) * align
        assert entry <= out.buffer_aligned_page_size(), (entry, out.buffer_aligned_page_size())
        return entry, elem_bytes
    return out.buffer_page_size(), elem_bytes


def default_block_pages(layout, entry_bytes, elem_bytes):
    """The block knob in pages: BLOCK_TILES tiles, or as many sticks as hold the same bytes."""
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        return max(1, BLOCK_TILES * TILE * TILE * elem_bytes // entry_bytes)
    return BLOCK_TILES


def split_tiles(num_tiles, grid_size):
    """(core set, {core: tiles}, {core: first tile}) over the clusters that get work, row-major."""
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid_size, num_tiles, row_wise=True
    )
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = per_core, start
            start += per_core
    return all_cores, counts, starts


def resolve_block_tiles(block_tiles, max_core_tiles, page_bytes):
    """The block extent in pages: the knob clamped by the ring cap, the L1 budget and the core's own share."""
    ring_cap_tiles = RING_MAX_BYTES // (OUTPUT_DEPTH * page_bytes)
    budget_tiles = (ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES) // (OUTPUT_DEPTH * page_bytes)
    return max(1, min(block_tiles, max_core_tiles, ring_cap_tiles, budget_tiles))


def _dm_config(make, explicit_rings):
    hw = make()
    if explicit_rings:
        hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=explicit_rings)
    return hw


def create_program_artifacts(tensors, out, dim, *, block_tiles=None):
    """(ProgramSpec, ProgramRunArgs, tensor indices) for concat of `tensors` along canonical `dim` into `out`.

    `block_tiles` is the block knob in output pages (tiles or sticks); None takes the layout's default."""
    count = len(tensors)
    assert 1 <= count <= MAX_INPUTS and dim in (-1, -2)
    shapes = [tuple(t.shape) for t in tensors]
    layout = out.layout
    concat_is_w = dim == -1
    cross = -2 if concat_is_w else -1
    regime = select_regime(shapes, dim, layout)

    extents = [int(s[dim]) for s in shapes]
    offsets = [sum(extents[:k]) for k in range(count)]
    num_images = 1
    for d in shapes[0][:-2]:
        num_images *= int(d)
    cross_extent = int(shapes[0][cross])
    cross_tiles = -(-cross_extent // TILE)
    concat_tiles_out = -(-sum(extents) // TILE)
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        total_out_pages = num_images * (int(shapes[0][-2]) if concat_is_w else sum(extents))  # one page per stick
    else:
        total_out_pages = num_images * cross_tiles * concat_tiles_out
    assert total_out_pages == out.buffer_num_pages(), (total_out_pages, out.buffer_num_pages())

    page_bytes, elem_bytes = page_geometry(out, layout)
    if block_tiles is None:
        block_tiles = default_block_pages(layout, page_bytes, elem_bytes)
    core_set, counts, starts = split_tiles(total_out_pages, out.device().compute_with_storage_grid_size())
    block = resolve_block_tiles(block_tiles, max(counts.values()), page_bytes)

    ring = E.DataflowBufferSpec(
        unique_id=RING, entry_size=page_bytes, num_entries=OUTPUT_DEPTH * block, data_format_metadata=out.dtype
    )
    names = [f"in_{k}" for k in range(count)]
    tensor_parameters = [E.TensorParameter(unique_id=n, spec=t.spec) for n, t in zip(names, tensors)]
    tensor_parameters.append(E.TensorParameter(unique_id="out", spec=out.spec))

    reader_explicit = [RING] if regime in EXPLICIT_REGIMES else []
    Schema = E.KernelSpec.RuntimeArgSchema
    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "concat_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, reader_explicit),
        num_threads=THREAD_PLAN["reader_threads"],
        dfb_bindings=[E.producer_of(RING, RING)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=n, accessor_name=n) for n in names],
        compile_time_args={
            "count": count,
            "concat_is_w": int(concat_is_w),
            "regime": REGIMES[regime],
            "block_tiles": block,
            "page_bytes": page_bytes,
            "elem_bytes": elem_bytes,
            "reader_threads": THREAD_PLAN["reader_threads"],
        },
        runtime_arg_schema=Schema(
            runtime_arg_names=["start_tile", "num_tiles"],
            common_runtime_arg_names=["cross_tiles", "cross_extent", "concat_tiles_out"],
            common_runtime_array_args=[
                Schema.ArrayArgDecl(name="piece_extent", size=count),
                Schema.ArrayArgDecl(name="piece_offset", size=count),
            ],
        ),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "concat_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config, []),
        num_threads=THREAD_PLAN["writer_threads"],
        dfb_bindings=[E.consumer_of(RING, RING)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="out", accessor_name="out")],
        compile_time_args={"writer_threads": THREAD_PLAN["writer_threads"]},
        runtime_arg_schema=Schema(runtime_arg_names=["start_tile", "num_tiles"]),
    )
    spec = E.ProgramSpec(
        name="concat",
        kernels=[reader, writer],
        dataflow_buffers=[ring],
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "writer"], target_nodes=core_set)],
    )
    dm_args = {"start_tile": starts, "num_tiles": counts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(
                kernel="reader",
                runtime_arg_values=dm_args,
                common_runtime_arg_values={
                    "cross_tiles": cross_tiles,
                    "cross_extent": cross_extent,
                    "concat_tiles_out": concat_tiles_out,
                },
                common_runtime_array_arg_values={"piece_extent": extents, "piece_offset": offsets},
            ),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
        ]
    )
    tensor_indices = {n: k for k, n in enumerate(names)}
    tensor_indices["out"] = count
    return spec, run_args, tensor_indices
