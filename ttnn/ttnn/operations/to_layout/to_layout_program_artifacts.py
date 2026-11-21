# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for to_layout (ttnn.ProgramSpec; Quasar runs nothing else).

Three regimes, each its own ProgramSpec; a call builds and dispatches exactly one (op_design.md, Regimes):

    tilize_stream    RM sticks -> cb_row_block -> tilize -> cb_tile_block -> TILE pages
    untilize_stream  TILE pages -> cb_tile_block_in -> untilize -> cb_row_block_out -> RM sticks
    identity_copy    pages -> cb_pages -> pages (data-movement only)

Blocking (op_design.md, Blocking Model): a unit of work is a band of `block_height_tiles` tile-rows by
`block_width_tiles` (bw) tile-columns. Leading dims fold into images; every image is tile-padded on its
own (Ht = ceil(H/32) per image). Units are flattened (row band major, column block minor) and spread over
the grid by `split_work_to_cores(row_wise=True)`: contiguous unit ranges per cluster.

Every knob is one host constant (or one host function) and reaches the kernels as one compile-time arg:
    BLOCK_HEIGHT_TILES   tile-rows per block                        (Phase 0: 1)
    select_block_width_tiles()  tile-columns per block              (occupancy first, then coarsest)
    BUFFER_DEPTH         blocks per ring                            (Phase 0: 2)
    COPY_BLOCK_PAGES     pages per implicit-sync half (identity)    (Phase 0: 4)
    THREAD_PLAN          (readers, NEOs, writers) per cluster       (Phase 0: (1, 1, 1))

Kernel-local ring names are by role, so each kernel source serves every regime: the reader produces into
`staged_input`, the writer drains `staged_output`; the compute kernel consumes `staged_input` and produces
`staged_output`. In identity_copy both names bind the one ring `cb_pages`.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# ---------------------------------------------------------------------------
# Knobs (single source of truth)
# ---------------------------------------------------------------------------

# Thread plan (R readers, C NEOs, W writers) per cluster. The op is data-movement bound (target page,
# "Choosing the plan"): one thread of each kind.
# The tilize / untilize unit is a whole tile-row block (bw ring entries), so a plan with C > 1
# would need BLOCKED rings (op_design.md, Work Distribution); validate_thread_plan refuses it until built.
THREAD_PLAN = (1, 1, 1)

BLOCK_HEIGHT_TILES = 1
BUFFER_DEPTH = 2
COPY_BLOCK_PAGES = 4

# Block-width selection constants (op_design.md, `block_width_tiles` selection).
MIN_CHUNK_BYTES = 256  # one NoC payload width on this part: the floor of a stick transfer
DFB_RING_MAX_BYTES = 1_048_560  # one ring's size limit (target page, Cluster model)
L1_RESERVE_BYTES = 1 << 20  # left free for L1-resident input / output tensors (R3 / R5)

TILE_HEIGHT = 32
TILE_WIDTH = 32

# The cluster model of the target page; Python has no query for them.
FREE_DM_HARTS = 6
NEOS = 4

REGIME_TILIZE = 0
REGIME_UNTILIZE = 1
REGIME_COPY = 2

# Implicit-sync opt-outs: the ring each regime's explicit side keeps the credits of itself.
#   tilize:   the reader assembles a block from many stick reads + zeroing -> explicit on cb_row_block
#   untilize: the writer scatters a block into many stick writes          -> explicit on cb_row_block_out


def validate_thread_plan(threads, regime):
    r, c, w = threads
    if not (1 <= c <= NEOS and r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and c % r == 0 and c % w == 0):
        raise ValueError(f"thread plan (R, C, W) = {threads}: need C <= {NEOS}, R + W <= {FREE_DM_HARTS}, R | C, W | C")
    if regime in (REGIME_TILIZE, REGIME_UNTILIZE) and threads != (1, 1, 1):
        # A tile-row block is bw entries: STRIDED dealing per entry would split it across NEOs. A plan with
        # more threads needs BLOCKED rings (block_size = bw) and whole rounds of C blocks per cluster.
        raise NotImplementedError(f"to_layout: thread plan {threads} needs BLOCKED rings, not built")
    if regime == REGIME_COPY and (r, w) != (1, 1):
        raise NotImplementedError(f"to_layout: identity copy with thread plan {threads} is not built")


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def select_block_width_tiles(total_row_bands, tensor_w_tiles, tile_bytes, elem_size, num_cores, dram_alignment, l1_budget):
    """bw: occupancy first, then the coarsest divisor of tensor_w_tiles within the caps (op_design.md)."""
    # 1. floor: a stick transfer of at least one NoC payload width.
    bw_min = min(tensor_w_tiles, math.ceil(MIN_CHUNK_BYTES / (TILE_WIDTH * elem_size)))
    # 2. caps: one ring, and both live rings within the per-cluster budget.
    ring_bytes_per_bw = BUFFER_DEPTH * BLOCK_HEIGHT_TILES * tile_bytes
    bw_max = min(DFB_RING_MAX_BYTES // ring_bytes_per_bw, max(1, l1_budget // (2 * ring_bytes_per_bw)))
    bw_max = max(1, bw_max)
    # 3. legal set: divisors (the Quasar tilize MOP width must equal the init width: no ragged column
    #    block) whose stick chunk keeps the DRAM alignment.
    divs = _divisors(tensor_w_tiles)
    legal = [
        d for d in divs if bw_min <= d <= bw_max and (d * TILE_WIDTH * elem_size) % dram_alignment == 0
    ]
    if not legal:
        legal = [max(d for d in divs if d <= bw_max)]
    # 4. occupancy first: the widest block that still gives every cluster a unit.
    filling = [d for d in legal if total_row_bands * (tensor_w_tiles // d) >= num_cores]
    return max(filling) if filling else min(legal)


def split_units(num_units, grid_size):
    """(core set, {core: units}, {core: first unit}) over the clusters that get work, row-major."""
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


def _regime(input_tensor, output_tensor):
    if input_tensor.layout == output_tensor.layout:
        return REGIME_COPY
    return REGIME_TILIZE if output_tensor.layout == ttnn.TILE_LAYOUT else REGIME_UNTILIZE


def block_geometry(input_tensor, output_tensor, device):
    """Every derived block quantity of a tilize / untilize call, from the knobs and the shape."""
    shape = tuple(input_tensor.shape)
    H, W = shape[-2], shape[-1]
    num_images = math.prod(shape[:-2]) if len(shape) > 2 else 1
    Ht = math.ceil(H / TILE_HEIGHT)
    tensor_w_tiles = math.ceil(W / TILE_WIDTH)
    total_tile_rows = num_images * Ht
    tile_tensor = output_tensor if output_tensor.layout == ttnn.TILE_LAYOUT else input_tensor
    tile_bytes = tile_tensor.buffer_page_size()
    elem_size = tile_tensor.element_size()
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    total_row_bands = math.ceil(total_tile_rows / BLOCK_HEIGHT_TILES)
    l1_budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES
    bw = select_block_width_tiles(
        total_row_bands, tensor_w_tiles, tile_bytes, elem_size, num_cores, ttnn.get_dram_alignment(), l1_budget
    )
    return {
        "H": H,
        "Ht": Ht,
        "tensor_w_tiles": tensor_w_tiles,
        "total_tile_rows": total_tile_rows,
        "total_row_bands": total_row_bands,
        "tile_bytes": tile_bytes,
        "elem_size": elem_size,
        "block_width_tiles": bw,
        "n_col_blocks": tensor_w_tiles // bw,
        "chunk_bytes": bw * TILE_WIDTH * elem_size,
        "grid": grid,
    }


def _tile_rows_of_units(unit_start, num_units, n_col_blocks, total_tile_rows):
    """Tile-rows the compute kernel tilizes / untilizes for a contiguous unit range."""
    rows = 0
    for u in range(unit_start, unit_start + num_units):
        band = u // n_col_blocks
        rows += min(BLOCK_HEIGHT_TILES, total_tile_rows - band * BLOCK_HEIGHT_TILES)
    return rows


def create_program_artifacts(input_tensor, output_tensor, *, threads=THREAD_PLAN):
    regime = _regime(input_tensor, output_tensor)
    validate_thread_plan(threads, regime)
    if regime == REGIME_COPY:
        return _copy_artifacts(input_tensor, output_tensor, threads)
    return _stream_artifacts(input_tensor, output_tensor, threads, regime)


def _tensor_parameters(input_tensor, output_tensor):
    return [
        E.TensorParameter(unique_id="input", spec=input_tensor.spec),
        E.TensorParameter(unique_id="output", spec=output_tensor.spec),
    ]


def _stream_artifacts(input_tensor, output_tensor, threads, regime):
    R, C, W = threads
    device = input_tensor.device()
    g = block_geometry(input_tensor, output_tensor, device)
    bw = g["block_width_tiles"]
    num_units = g["total_row_bands"] * g["n_col_blocks"]
    core_set, counts, starts = split_units(num_units, g["grid"])
    tile_rows = {
        core: _tile_rows_of_units(starts[core], counts[core], g["n_col_blocks"], g["total_tile_rows"]) for core in counts
    }

    if regime == REGIME_TILIZE:
        in_ring, out_ring = "cb_row_block", "cb_tile_block"
        reader_explicit, writer_explicit = [in_ring], []
        name = "to_layout_tilize"
    else:
        in_ring, out_ring = "cb_tile_block_in", "cb_row_block_out"
        reader_explicit, writer_explicit = [], [out_ring]
        name = "to_layout_untilize"

    # Capacity per NEO x C: BUFFER_DEPTH blocks of block_height_tiles x bw tile-sized entries.
    entries = BUFFER_DEPTH * BLOCK_HEIGHT_TILES * bw * C
    dfbs = [
        E.DataflowBufferSpec(
            unique_id=ring, entry_size=g["tile_bytes"], num_entries=entries, data_format_metadata=input_tensor.dtype
        )
        for ring in (in_ring, out_ring)
    ]

    plan = {"reader_threads": R, "compute_threads": C, "writer_threads": W}
    geometry_args = {
        "regime": regime,
        "block_width_tiles": bw,
        "block_height_tiles": BLOCK_HEIGHT_TILES,
        "n_col_blocks": g["n_col_blocks"],
        "tensor_w_tiles": g["tensor_w_tiles"],
        "total_tile_rows": g["total_tile_rows"],
        "image_height": g["H"],
        "image_height_tiles": g["Ht"],
        "chunk_bytes": g["chunk_bytes"],
    }
    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["unit_start", "num_units"])

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "to_layout_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, reader_explicit),
        num_threads=R,
        dfb_bindings=[E.producer_of(in_ring, "staged_input")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args={**plan, **geometry_args},
        runtime_arg_schema=dm_schema,
    )
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "to_layout_compute.cpp"),
        # A datacopy has no multiply, so the fidelity cannot change a value; it is always set explicitly.
        hw_config=E.ComputeHardwareConfig(fpu_math_fidelity=ttnn.MathFidelity.LoFi, enable_32_bit_dest=False),
        num_threads=C,
        dfb_bindings=[E.consumer_of(in_ring, "staged_input"), E.producer_of(out_ring, "staged_output")],
        compile_time_args={**plan, "regime": regime, "block_width_tiles": bw},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tile_rows"]),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "to_layout_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config, writer_explicit),
        num_threads=W,
        dfb_bindings=[E.consumer_of(out_ring, "staged_output")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args={**plan, **geometry_args},
        runtime_arg_schema=dm_schema,
    )
    spec = E.ProgramSpec(
        name=name,
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=_tensor_parameters(input_tensor, output_tensor),
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=core_set)],
    )
    dm_args = {"unit_start": starts, "num_units": counts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="compute", runtime_arg_values={"num_tile_rows": tile_rows}),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {"input": 0, "output": 1}


def _copy_artifacts(input_tensor, output_tensor, threads):
    R, C, W = threads
    device = input_tensor.device()
    num_pages = input_tensor.buffer_num_pages()
    page_bytes = input_tensor.buffer_page_size()
    core_set, counts, starts = split_units(num_pages, device.compute_with_storage_grid_size())

    # DM -> DM ring: the bytes never pass DEST, so no format metadata.
    dfbs = [E.DataflowBufferSpec(unique_id="cb_pages", entry_size=page_bytes, num_entries=BUFFER_DEPTH * COPY_BLOCK_PAGES)]
    plan = {"reader_threads": R, "compute_threads": C, "writer_threads": W}
    ct_args = {**plan, "regime": REGIME_COPY}
    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["unit_start", "num_units"])
    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "to_layout_copy_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=R,
        dfb_bindings=[E.producer_of("cb_pages", "staged_input")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args=ct_args,
        runtime_arg_schema=dm_schema,
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "to_layout_copy_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        dfb_bindings=[E.consumer_of("cb_pages", "staged_output")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args=ct_args,
        runtime_arg_schema=dm_schema,
    )
    spec = E.ProgramSpec(
        name="to_layout_copy",
        kernels=[reader, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=_tensor_parameters(input_tensor, output_tensor),
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "writer"], target_nodes=core_set)],
    )
    dm_args = {"unit_start": starts, "num_units": counts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {"input": 0, "output": 1}
