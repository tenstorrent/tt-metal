# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for split (op_design.md, Blocking Model).

One program writes every chunk. Data-movement kernels only (thread plan (R, C, W) = (1, 0, 1)): a reader hart
fills the ring `cb_output_pages` with output pages, a writer hart drains each entry into the page of the chunk
it belongs to. Two regimes, chosen by `select_regime` from the shape and `split_size` alone (never the grid):

page_remap      every output page is exactly one input page (ROW_MAJOR cut along a page-row dim; TILE cut
                before H, or along H at a multiple of the tile height). Reader and writer both implicit sync.
tile_row_shift  TILE cut along H inside a tile: an output tile takes its rows from up to two input tiles. The
                reader keeps a window of input tiles in a scratchpad (one DRAM barrier per block of
                `BLOCK_H_TILES` output tiles, which may span tile columns) and assembles each output tile into the ring with self-aimed NoC
                copies (explicit sync, every transfer trid 0). The writer stays implicit.

The geometry of a call is derived once here (`derive_geometry`) and passed to both kernels as common runtime
args; kernels/split_walk.hpp maps a unit of the walk to its input and output pages the same way on both sides.
"""

from math import ceil, prod
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# ---------------------------------------------------------------------------
# Knobs (each defined once; everything dependent derives from them)
# ---------------------------------------------------------------------------

# Thread plan (R readers, C compute threads, W writers) per cluster. No compute kernel: the op only moves data.
THREAD_PLAN = (1, 0, 1)
# Entries of cb_output_pages per DM thread: reader fills entry k+1 while the writer drains entry k.
RING_DEPTH = 2
# tile_row_shift: output tiles of the walk assembled per DRAM-read barrier (a block may span tile columns).
BLOCK_H_TILES = 8
# tile_row_shift: input tiles resident in the scratchpad window. A column segment of m output tiles reads at most
# m + 1 input tiles, so a block of BLOCK_H_TILES output tiles reads at most 2 * BLOCK_H_TILES (shared tile included).
WINDOW_TILES = 2 * BLOCK_H_TILES
# The writer binds one accessor per chunk tensor (chunk_0 .. chunk_{MAX_CHUNKS-1}); passed to it as the define
# SPLIT_MAX_CHUNKS, so the cap is stated once.
MAX_CHUNKS = 4
# The tile the tile_row_shift assembly addresses (four 16x16 faces; split_walk.hpp TILE_ROWS = its height).
ASSEMBLY_TILE = (32, 32)

# Mechanism cap: an implicit-sync transfer is one ring entry, one NoC transaction (no size argument).
NOC_MAX_BURST_BYTES = 256 * 256  # NOC_MAX_BURST_WORDS x NOC_WORD_BYTES on quasar (noc_parameters.h:395-396)

# The cluster model of the target page; Python has no query for it.
FREE_DM_HARTS = 6

# Regime ids, shared with kernels/split_walk.hpp (split::Regime).
PAGE_REMAP = "page_remap"
TILE_ROW_SHIFT = "tile_row_shift"
REGIME_IDS = {PAGE_REMAP: 0, TILE_ROW_SHIFT: 1}
DEFERRED_REGIMES = ("tile_col_shift", "rm_substick")


def select_regime(layout, dim, split_size, tile_shape=(32, 32)):
    """The design's regime-selection function: exact, from the layout, canonical dim and split_size only."""
    tile_h, tile_w = tile_shape
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        return "rm_substick" if dim == -1 else PAGE_REMAP
    if dim <= -3:
        return PAGE_REMAP
    if split_size % (tile_h if dim == -2 else tile_w) == 0:
        return PAGE_REMAP
    return TILE_ROW_SHIFT if dim == -2 else "tile_col_shift"


def validate_thread_plan(threads):
    r, c, w = threads
    if c != 0:
        raise ValueError(f"split runs no compute kernel: thread plan {threads} must have C = 0")
    if not (r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS):
        raise ValueError(f"thread plan (R, C, W) = {threads}: need R, W >= 1 and R + W <= {FREE_DM_HARTS}")
    # A DM -> DM ring deals entries round-robin to the producer threads and to the consumer threads; a reader
    # and a writer thread must therefore see the same entries, which holds when R == W. The tile_row_shift
    # reader is a single hart (its window is per hart; dealing columns to two readers is a perf lamp).
    if r != w:
        raise ValueError(f"thread plan {threads}: R must equal W on the DM -> DM ring")


def _tile_shape(tensor):
    tile = tensor.tile.tile_shape if hasattr(tensor, "tile") and tensor.tile is not None else (32, 32)
    return int(tile[0]), int(tile[1])


def derive_geometry(input_tensor, split_size, dim):
    """The single host derivation of every geometry quantity (op_design.md, Canonical page view).

    Returns (regime, n_chunks, num_units, geometry dict of the kernels' common runtime args).
    """
    shape = [int(s) for s in input_tensor.shape]
    layout = input_tensor.layout
    tile_h, tile_w = _tile_shape(input_tensor)
    regime = select_regime(layout, dim, split_size, (tile_h, tile_w))
    if regime in DEFERRED_REGIMES:
        raise NotImplementedError(f"split: regime {regime} (dim {dim}, layout {layout}) is not built yet")

    extent = shape[dim]
    n_chunks = ceil(extent / split_size)
    if n_chunks > MAX_CHUNKS:
        raise NotImplementedError(f"split: {n_chunks} chunks; the writer binds at most {MAX_CHUNKS}")

    # Fields both regimes carry (unused ones are 0), so the kernels' arg schema is one list.
    geometry = dict.fromkeys(
        ("cut_pages", "inner_pages", "chunk_pages", "h_tiles", "w_tiles", "split_rows", "cut_rows", "chunk_tiles",
         "column_tiles"),
        0,
    )
    if regime == PAGE_REMAP:
        if layout == ttnn.ROW_MAJOR_LAYOUT:
            # A page is a stick: page-rows along every dim before W.
            outer = prod(shape[:dim])
            cut_pages = extent
            inner_pages = prod(shape[dim + 1 : -1])
            chunk_pages = split_size
        elif dim == -2:
            outer = prod(shape[:-2])
            cut_pages = ceil(shape[-2] / tile_h)
            inner_pages = ceil(shape[-1] / tile_w)
            chunk_pages = split_size // tile_h
        elif dim == -1:
            # TILE cut along W at a multiple of the tile width: every tile-row of every image is a page-row,
            # cut into tile columns (op_design.md, Regimes, rm_substick row's note).
            outer = prod(shape[:-2]) * ceil(shape[-2] / tile_h)
            cut_pages = ceil(shape[-1] / tile_w)
            inner_pages = 1
            chunk_pages = split_size // tile_w
        else:  # TILE, dim <= -3: whole images of tiles
            outer = prod(shape[:dim])
            cut_pages = extent
            inner_pages = prod(shape[dim + 1 : -2]) * ceil(shape[-2] / tile_h) * ceil(shape[-1] / tile_w)
            chunk_pages = split_size
        geometry.update(cut_pages=cut_pages, inner_pages=inner_pages, chunk_pages=chunk_pages)
        num_units = outer * cut_pages * inner_pages
    else:  # TILE_ROW_SHIFT: walk (o, tw, c, j) over output tiles, column-major
        if (tile_h, tile_w) != ASSEMBLY_TILE:
            # The row-run assembly (split_reader.cpp, split_walk.hpp TILE_ROWS) addresses four 16x16 faces.
            raise NotImplementedError(f"split: tile_row_shift assembles {ASSEMBLY_TILE} tiles, not {(tile_h, tile_w)}")
        outer = prod(shape[:-2])
        h_tiles = ceil(shape[-2] / tile_h)
        w_tiles = ceil(shape[-1] / tile_w)
        chunk_tiles = ceil(split_size / tile_h)  # output tile-rows of a full chunk
        last_rows = extent - (n_chunks - 1) * split_size
        column_tiles = (n_chunks - 1) * chunk_tiles + ceil(last_rows / tile_h)  # output tiles per input column
        geometry.update(
            h_tiles=h_tiles,
            w_tiles=w_tiles,
            split_rows=split_size,
            cut_rows=extent,
            chunk_tiles=chunk_tiles,
            column_tiles=column_tiles,
        )
        num_units = outer * w_tiles * column_tiles
    return regime, n_chunks, num_units, geometry


def split_units(num_units, grid_size):
    """(core set, {core: units}, {core: first unit}) over the clusters that get work: contiguous walk ranges."""
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid_size, num_units, row_wise=True
    )
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = per_core, start
            start += per_core
    return all_cores, counts, starts


def _dm_config(make, explicit_rings):
    hw = make()
    if explicit_rings:
        hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=explicit_rings)
    return hw


def create_program_artifacts(input_tensor, chunks, split_size, dim, *, threads=THREAD_PLAN):
    """(ProgramSpec, ProgramRunArgs, tensor_args) for one dispatch that writes every chunk."""
    validate_thread_plan(threads)
    R, _, W = threads
    regime, n_chunks, num_units, geometry = derive_geometry(input_tensor, split_size, dim)
    if len(chunks) != n_chunks:
        raise ValueError(f"split: {len(chunks)} chunk tensors for {n_chunks} chunks")
    if regime == TILE_ROW_SHIFT and R != 1:
        raise ValueError(f"split: the tile_row_shift reader is one hart (thread plan {threads})")

    page_bytes = int(input_tensor.buffer_aligned_page_size())
    if page_bytes > NOC_MAX_BURST_BYTES:
        raise NotImplementedError(f"split: page of {page_bytes} B exceeds one NoC burst ({NOC_MAX_BURST_BYTES} B)")
    for k, chunk in enumerate(chunks):
        # A ring entry is written whole into a chunk page: a larger page would spill into its neighbour.
        if int(chunk.buffer_aligned_page_size()) != page_bytes:
            raise ValueError(f"split: chunk {k} page {chunk.buffer_aligned_page_size()} B != input page {page_bytes} B")
    if regime == TILE_ROW_SHIFT:
        face_row_bytes = page_bytes // 64
        if face_row_bytes % ttnn.get_l1_alignment() != 0:
            raise NotImplementedError(f"split: face row of {face_row_bytes} B is not L1-aligned")

    core_set, counts, starts = split_units(num_units, input_tensor.device().compute_with_storage_grid_size())

    ring = E.DataflowBufferSpec(
        unique_id="cb_output_pages",
        entry_size=page_bytes,
        num_entries=RING_DEPTH * R,
        data_format_metadata=input_tensor.dtype,
    )
    scratchpads = []
    reader_scratch = []
    if regime == TILE_ROW_SHIFT:
        # DM-only scratchpad: no format metadata.
        scratchpads.append(E.ScratchpadSpec(unique_id="scratch_input_window", size_per_node=WINDOW_TILES * page_bytes))
        reader_scratch.append(
            E.ScratchpadBinding(scratchpad_spec_name="scratch_input_window", accessor_name="scratch_input_window")
        )

    chunk_names = [f"chunk_{k}" for k in range(n_chunks)]
    tensor_parameters = [E.TensorParameter(unique_id="input", spec=input_tensor.spec)] + [
        E.TensorParameter(unique_id=name, spec=chunk.spec) for name, chunk in zip(chunk_names, chunks)
    ]

    plan = {"reader_threads": R, "writer_threads": W}
    reader_ct = {**plan, "block_h_tiles": BLOCK_H_TILES, "window_tiles": WINDOW_TILES}
    # The regime is a define (kernels/split_walk.hpp says why); the chunk count names the writer's accessors.
    regime_define = {"SPLIT_REGIME": str(REGIME_IDS[regime])}
    geometry_names = list(geometry)
    schema = E.KernelSpec.RuntimeArgSchema(
        runtime_arg_names=["start_unit", "num_units"], common_runtime_arg_names=geometry_names
    )

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "split_reader.cpp"),
        hw_config=_dm_config(
            E.create_reader_datamovement_config, ["cb_output_pages"] if regime == TILE_ROW_SHIFT else []
        ),
        num_threads=R,
        compiler_options=E.KernelSpec.CompilerOptions(defines=regime_define),
        dfb_bindings=[E.producer_of("cb_output_pages", "cb_output_pages")],
        scratchpad_bindings=reader_scratch,
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args=reader_ct,
        runtime_arg_schema=schema,
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "split_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config, []),
        num_threads=W,
        compiler_options=E.KernelSpec.CompilerOptions(defines={**regime_define, "SPLIT_NUM_CHUNKS": str(n_chunks), "SPLIT_MAX_CHUNKS": str(MAX_CHUNKS)}),
        dfb_bindings=[E.consumer_of("cb_output_pages", "cb_output_pages")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=name, accessor_name=name) for name in chunk_names],
        compile_time_args=plan,
        runtime_arg_schema=schema,
    )
    spec = E.ProgramSpec(
        name="split",
        kernels=[reader, writer],
        dataflow_buffers=[ring],
        scratchpads=scratchpads,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "writer"], target_nodes=core_set)],
    )
    per_core = {"start_unit": starts, "num_units": counts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel=k, runtime_arg_values=per_core, common_runtime_arg_values=dict(geometry))
            for k in ("reader", "writer")
        ]
    )
    tensor_args = {"input": 0, **{name: 1 + k for k, name in enumerate(chunk_names)}}
    return spec, run_args, tensor_args
