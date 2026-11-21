# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts of reshape (`ttnn.ProgramSpec`, one `ttnn.generic_op` per call).

Four regimes (op_design.md, Blocking Model -> Regimes); each is its own ProgramSpec, selected on the host:

page_identity      output page p is input page p (row-major with W kept; tile layout when the tiles hold the same
                   rows before and after). Two DM kernels on every engaged cluster: the reader moves a contiguous
                   page range into `dfb_pages`, the writer moves it out. Both ring sides use implicit sync.

tile_row_regroup   tile layout, H changes and not both H's are multiples of 32: rows move between tiles. Three DM
                   kernels: the reader assembles a window of whole input tiles (explicit sync), the relayout
                   (`reshape_compute.cpp`, a DM kernel: the op computes nothing) copies face-row runs from the
                   window into each output tile, the writer sends the output tiles (implicit sync).

w_regroup_rm       row-major, W changes: output row j is bytes [j*page_out, (j+1)*page_out) of the flat byte stream
                   of the input rows. The topology of tile_row_regroup on a one-column grid of row pages: the same
                   window reader (whole input pages, explicit sync), the relayout copying byte runs from the window's
                   input pages into each output page, and regime A's page writer (implicit sync).

w_regroup_tile     tile layout, W changes: output row g is flat elements [g*w_out, (g+1)*w_out), so elements move
                   between tiles inside face rows. The topology of tile_row_regroup: the same window reader (input
                   tile-rows x a per-cluster input tile-column window), the relayout copying element runs (cut at input
                   row, input face-row and output face-row ends) into each output tile, and the same tile writer.

Every block extent, buffer depth and core assignment is a parameter computed here once and passed to the kernels
as a compile-time or runtime argument; the kernels never divide (the Quasar XIP DM build breaks on libgcc divides).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# ---------------------------------------------------------------------------------------------------------------
# Knobs (op_design.md: Blocking Model, Buffer-depth knobs, Work Distribution). Each is defined once, here.
# ---------------------------------------------------------------------------------------------------------------

# The thread plan (R, relayout, W): one hart per DM kernel. The op has no compute kernel, so no NEO is used.
# Regime B's reader assembles a contiguous window and its relayout owns that window, so both must stay at 1;
# the regime-A reader and writer and the regime-B writer stride their loops by their count.
THREAD_PLAN = {"reader_threads": 1, "relayout_threads": 1, "writer_threads": 1}
FREE_DM_HARTS = 6  # target page, Cluster model: 6 of the 8 DM cores are for user kernels

BLOCK_BYTES_TARGET = 32 * 1024  # bytes per block of pages (regime A) / per output quantum (regime B)
PAGE_RING_DEPTH = 2  # dfb_pages = PAGE_RING_DEPTH x block_pages
OUTPUT_RING_DEPTH = 2  # dfb_output_tiles = OUTPUT_RING_DEPTH x output_quantum
L1_RESERVE_BYTES = 1024 * 1024  # headroom for L1-resident input/output tensors sharing the cluster's L1
DFB_RING_MAX_BYTES = 1_048_560  # target page, Cluster model: one ring <= 1 MiB - 16 B
NOC_MAX_BURST_SIZE = 64 * 1024  # one NoC transaction (quasar noc_parameters.h): a row-major page must fit
UINT16_MAX = 0xFFFF  # reserve_back / wait_front / push_back / pop_front take uint16_t counts (dataflow_buffer.h)

# Every ring is a byte ring between DM kernels: no compute kernel unpacks or packs it, and the kernels never read a
# ring's format. So no ring carries a data-format tag (a format-less DFB, which the runtime skips when it sets the
# compute JIT's CB formats). Bytes move verbatim whatever the dtype (rule R1), and the op does not depend on the
# part's format table (Quasar has no UInt32 entry, yet a uint32 tensor moves like any other 4-byte dtype).
RING_DATA_FORMAT = None

# Tile geometry: 32 x 32, four 16 x 16 faces (validated against the tensor's tile on the host). The single source:
# the relayout kernel receives `tile_rows` / `face_rows` as compile-time arguments.
TILE_ROWS = 32
FACE_ROWS = TILE_ROWS // 2
FACES_PER_TILE = 4
FACE_ROWS_PER_TILE = FACES_PER_TILE * FACE_ROWS

REGIME_PAGE_IDENTITY = "page_identity"
REGIME_TILE_ROW_REGROUP = "tile_row_regroup"
REGIME_W_REGROUP_RM = "w_regroup_rm"
REGIME_W_REGROUP_TILE = "w_regroup_tile"


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _hw(shape):
    """(H, W) of a shape; a rank-1 shape has H = 1 (the specification's convention)."""
    return (shape[-2] if len(shape) > 1 else 1, shape[-1])


def validate_thread_plan(plan=THREAD_PLAN):
    r, m, w = plan["reader_threads"], plan["relayout_threads"], plan["writer_threads"]
    if not (r >= 1 and w >= 1 and m >= 1 and r + w + m <= FREE_DM_HARTS):
        raise ValueError(f"reshape thread plan {plan}: needs >= 1 thread per kernel and <= {FREE_DM_HARTS} DM harts")


def select_regime(in_shape, out_shape, layout):
    """The regime-selection function of op_design.md."""
    h_in, w_in = _hw(in_shape)
    h_out, w_out = _hw(out_shape)
    if w_in != w_out:
        return REGIME_W_REGROUP_RM if layout == ttnn.ROW_MAJOR_LAYOUT else REGIME_W_REGROUP_TILE
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        return REGIME_PAGE_IDENTITY
    if h_in == h_out or (h_in % TILE_ROWS == 0 and h_out % TILE_ROWS == 0):
        return REGIME_PAGE_IDENTITY
    return REGIME_TILE_ROW_REGROUP


def _balanced_ranges(total, groups):
    """`groups` contiguous (start, count) ranges over [0, total) whose counts differ by at most 1."""
    base, rem = divmod(total, groups)
    ranges, start = [], 0
    for g in range(groups):
        count = base + (1 if g < rem else 0)
        ranges.append((start, count))
        start += count
    return ranges


def _grid_cores(grid, n):
    """The first n cores of the grid, row-major (x fastest), as CoreCoords."""
    return [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]


def _core_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _dm_config(make, explicit_rings=None, all_explicit=False):
    hw = make(disable_dfb_implicit_sync_for_all=all_explicit)
    if explicit_rings:
        hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=explicit_rings)
    return hw


# ---------------------------------------------------------------------------------------------------------------
# Regime A: page_identity
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class PageIdentityPlan:
    page_bytes: int
    entry_bytes: int
    num_pages: int
    block_pages: int
    ring_entries: int
    cores: list
    core_pages: dict = field(default_factory=dict)
    start_page: dict = field(default_factory=dict)


def plan_page_identity(input_tensor, output_tensor, *, block_bytes_target=BLOCK_BYTES_TARGET, plan=THREAD_PLAN):
    num_pages = input_tensor.buffer_num_pages()
    page_bytes = input_tensor.buffer_page_size()
    if num_pages != output_tensor.buffer_num_pages() or page_bytes != output_tensor.buffer_page_size():
        raise RuntimeError(
            f"reshape page_identity: input ({num_pages} x {page_bytes} B) and output "
            f"({output_tensor.buffer_num_pages()} x {output_tensor.buffer_page_size()} B) pages differ"
        )
    entry_bytes = _round_up(page_bytes, ttnn.get_l1_alignment())
    grid = input_tensor.device().compute_with_storage_grid_size()
    _, _, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, num_pages, row_wise=True)
    cores, core_pages, start_page, start = [], {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            cores.append(core)
            core_pages[core], start_page[core] = per_core, start
            start += per_core
    max_core_pages = max(core_pages.values())
    block_pages = max(1, min(block_bytes_target // entry_bytes, max_core_pages))
    # Implicit sync splits a side's batch evenly over its threads: keep the block a multiple of each thread count.
    block_pages = _round_up(block_pages, math.lcm(plan["reader_threads"], plan["writer_threads"]))
    return PageIdentityPlan(
        page_bytes=page_bytes,
        entry_bytes=entry_bytes,
        num_pages=num_pages,
        block_pages=block_pages,
        ring_entries=PAGE_RING_DEPTH * block_pages,
        cores=cores,
        core_pages=core_pages,
        start_page=start_page,
    )


def _page_identity_artifacts(input_tensor, output_tensor, p: PageIdentityPlan, plan):
    dfb = E.DataflowBufferSpec(
        unique_id="pages", entry_size=p.entry_bytes, num_entries=p.ring_entries, data_format_metadata=RING_DATA_FORMAT
    )
    ct = {**plan, "block_pages": p.block_pages}
    schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["core_pages", "start_page"])
    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "reshape_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config),
        num_threads=plan["reader_threads"],
        dfb_bindings=[E.producer_of("pages", "in_ring")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args=ct,
        runtime_arg_schema=schema,
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "reshape_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config),
        num_threads=plan["writer_threads"],
        dfb_bindings=[E.consumer_of("pages", "out_ring")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args=ct,
        runtime_arg_schema=schema,
    )
    spec = E.ProgramSpec(
        name="reshape_page_identity",
        kernels=[reader, writer],
        dataflow_buffers=[dfb],
        tensor_parameters=[
            E.TensorParameter(unique_id="input", spec=input_tensor.spec),
            E.TensorParameter(unique_id="output", spec=output_tensor.spec),
        ],
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "writer"], target_nodes=_core_set(p.cores))],
    )
    args = {"core_pages": p.core_pages, "start_page": p.start_page}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=args),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=args),
        ]
    )
    return spec, run_args


# ---------------------------------------------------------------------------------------------------------------
# The window reader (regimes tile_row_regroup and w_regroup_rm)
# ---------------------------------------------------------------------------------------------------------------


def _window_reader_spec(ct, max_row_blocks, plan):
    """load_window_block: for each block, input page-rows [window_first, window_first + window_rows) x the block's
    columns into the `input_window` ring, pushed at the nominal `window_entries` (explicit sync). Regime B reads a
    grid of tiles; regime W reads a one-column grid of row-major pages (`tensor_w_tiles = block_tile_cols = 1`)."""
    return E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "reshape_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, ["input_window"]),
        num_threads=plan["reader_threads"],
        dfb_bindings=[E.producer_of("input_window", "in_ring")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args=ct,
        compiler_options=E.KernelSpec.CompilerOptions(defines={"RESHAPE_WINDOW_READER": "1"}),
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["col_start", "col_count", "num_row_blocks"],
            runtime_array_args=[
                E.ArrayArgDecl(name="window_first", size=max_row_blocks),
                E.ArrayArgDecl(name="window_rows", size=max_row_blocks),
            ],
        ),
    )


def _tile_writer_spec(ct, plan):
    """store_tiles_block: the output tiles in the order the relayout pushes them (column block outer, row block inner,
    then output tile-row, then column), one implicit-sync write each, page k * tensor_w_tiles + w (regimes
    tile_row_regroup and w_regroup_tile; `ct` gives the output's tensor_w_tiles and the column / row block extents)."""
    return E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "reshape_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config),
        num_threads=plan["writer_threads"],
        dfb_bindings=[E.consumer_of("output_tiles", "out_ring")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args=ct,
        compiler_options=E.KernelSpec.CompilerOptions(defines={"RESHAPE_TILE_WRITER": "1"}),
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["col_start", "col_count", "row_start", "row_count"]
        ),
    )


# ---------------------------------------------------------------------------------------------------------------
# Regime B: tile_row_regroup
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class TileRowRegroupPlan:
    page_bytes: int
    h_in: int
    h_out: int
    tensor_w_tiles: int
    in_plane_tile_rows: int
    out_plane_tile_rows: int
    tensor_in_tile_rows: int
    tensor_out_tile_rows: int
    col_groups: int
    row_groups: int
    block_tile_cols: int
    block_out_tile_rows: int
    n_in_cap: int
    window_entries: int
    input_window_depth: int
    output_quantum: int
    max_row_blocks: int
    max_blocks_per_core: int
    cores: list
    # per core
    col_start: dict = field(default_factory=dict)
    col_count: dict = field(default_factory=dict)
    row_start: dict = field(default_factory=dict)
    row_count: dict = field(default_factory=dict)
    window_first: dict = field(default_factory=dict)  # core -> [per row block]
    window_rows: dict = field(default_factory=dict)  # core -> [per row block] (n_in)
    cursor: dict = field(default_factory=dict)  # core -> (in_tile_row, row_in_tile, row_in_plane, out_row_in_plane)

    def input_tiles_read(self):
        """Input tiles the readers fetch in total (a structural count: the minimum is in_tile_rows x w_tiles)."""
        total = 0
        for core in self.cores:
            col_blocks = -(-self.col_count[core] // self.block_tile_cols)
            total += sum(self.window_rows[core]) * self.col_count[core] if col_blocks else 0
        return total


class _RowGeometry:
    """Global logical row g <-> input/output tile-rows (op_design.md, n_in_cap definition)."""

    def __init__(self, h_in, h_out):
        self.h_in, self.h_out = h_in, h_out
        self.pin = -(-h_in // TILE_ROWS)
        self.pout = -(-h_out // TILE_ROWS)

    def in_tile_row_of(self, g):
        return (g // self.h_in) * self.pin + (g % self.h_in) // TILE_ROWS

    def first_global_row(self, k):
        return (k // self.pout) * self.h_out + TILE_ROWS * (k % self.pout)

    def valid_rows(self, k):
        return min(TILE_ROWS, self.h_out - TILE_ROWS * (k % self.pout))

    def window(self, ka, kb):
        """(first input tile-row, number of input tile-rows) feeding output tile-rows [ka, kb)."""
        first = self.in_tile_row_of(self.first_global_row(ka))
        last_row = self.first_global_row(kb - 1) + self.valid_rows(kb - 1) - 1
        return first, self.in_tile_row_of(last_row) - first + 1


def _row_blocks(row_start, row_count, block_out_tile_rows):
    return [(ka, min(ka + block_out_tile_rows, row_start + row_count)) for ka in range(row_start, row_start + row_count, block_out_tile_rows)]


def plan_tile_row_regroup(
    input_tensor, in_shape, out_shape, *, block_tile_cols=None, block_out_tile_rows=None, row_groups=None
):
    """The split and block extents of regime B. The keyword knobs override the Phase 0 values (whole per-core
    assignment, grid-filling row groups) for tests and refinements; None keeps the default."""
    page_bytes = input_tensor.buffer_page_size()
    h_in, w = _hw(in_shape)
    h_out, _ = _hw(out_shape)
    volume = math.prod(in_shape)
    geo = _RowGeometry(h_in, h_out)
    w_tiles = -(-w // TILE_ROWS)
    t_in = (volume // (h_in * w)) * geo.pin
    t_out = (volume // (h_out * w)) * geo.pout

    grid = input_tensor.device().compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    col_groups = min(w_tiles, num_cores)
    if row_groups is None:
        row_groups = max(1, min(t_out, t_in, num_cores // col_groups))
    row_groups = max(1, min(row_groups, t_out, num_cores // col_groups))
    col_ranges = _balanced_ranges(w_tiles, col_groups)
    row_ranges = _balanced_ranges(t_out, row_groups)
    max_core_cols = max(c for _, c in col_ranges)
    max_core_rows = max(c for _, c in row_ranges)

    budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES

    def n_in_cap_for(bh):
        return max(geo.window(ka, kb)[1] for rs, rc in row_ranges for ka, kb in _row_blocks(rs, rc, bh))

    def blocks_per_core(bh, wb):
        return -(-max_core_cols // wb) * -(-max_core_rows // bh)

    def footprint(bh, wb):
        depth = 2 if blocks_per_core(bh, wb) > 1 else 1
        window = depth * n_in_cap_for(bh) * wb * page_bytes
        quantum = min(wb, max(1, BLOCK_BYTES_TARGET // page_bytes))
        output = OUTPUT_RING_DEPTH * quantum * page_bytes
        return window, output

    def fits(bh, wb):
        window, output = footprint(bh, wb)
        return window <= DFB_RING_MAX_BYTES and output <= DFB_RING_MAX_BYTES and window + output <= budget

    # Fit search (l1_ledger.md): the whole per-core assignment first; shrink the column block (no traffic cost),
    # then the row block (one more boundary input tile-row re-read per step).
    bh = block_out_tile_rows if block_out_tile_rows is not None else max_core_rows
    wb = block_tile_cols if block_tile_cols is not None else max_core_cols
    bh, wb = max(1, min(bh, max_core_rows)), max(1, min(wb, max_core_cols))
    while not fits(bh, wb) and wb > 1:
        wb -= 1
    while not fits(bh, wb) and bh > 1:
        bh -= 1
    if not fits(bh, wb):
        raise NotImplementedError(f"reshape: no block of the tile-row regroup fits L1 for {in_shape} -> {out_shape}")

    n_in_cap = n_in_cap_for(bh)
    max_blocks = blocks_per_core(bh, wb)
    p = TileRowRegroupPlan(
        page_bytes=page_bytes,
        h_in=h_in,
        h_out=h_out,
        tensor_w_tiles=w_tiles,
        in_plane_tile_rows=geo.pin,
        out_plane_tile_rows=geo.pout,
        tensor_in_tile_rows=t_in,
        tensor_out_tile_rows=t_out,
        col_groups=col_groups,
        row_groups=row_groups,
        block_tile_cols=wb,
        block_out_tile_rows=bh,
        n_in_cap=n_in_cap,
        window_entries=n_in_cap * wb,
        input_window_depth=2 if max_blocks > 1 else 1,
        output_quantum=min(wb, max(1, BLOCK_BYTES_TARGET // page_bytes)),
        max_row_blocks=-(-max_core_rows // bh),
        max_blocks_per_core=max_blocks,
        cores=[],
    )
    cores = _grid_cores(grid, row_groups * col_groups)
    for rg, (rs, rc) in enumerate(row_ranges):
        g0 = geo.first_global_row(rs)
        r_in = g0 % h_in
        cursor = (geo.in_tile_row_of(g0), r_in % TILE_ROWS, r_in, TILE_ROWS * (rs % geo.pout))
        blocks = [geo.window(ka, kb) for ka, kb in _row_blocks(rs, rc, bh)]
        firsts = [f for f, _ in blocks] + [0] * (p.max_row_blocks - len(blocks))
        rows = [n for _, n in blocks] + [0] * (p.max_row_blocks - len(blocks))
        for cg, (cs, cc) in enumerate(col_ranges):
            core = cores[rg * col_groups + cg]
            p.cores.append(core)
            p.col_start[core], p.col_count[core] = cs, cc
            p.row_start[core], p.row_count[core] = rs, rc
            p.window_first[core], p.window_rows[core] = firsts, rows
            p.cursor[core] = cursor
    return p


def _tile_row_regroup_artifacts(input_tensor, output_tensor, p: TileRowRegroupPlan, plan):
    if plan["reader_threads"] != 1 or plan["relayout_threads"] != 1:
        raise ValueError("reshape tile_row_regroup: the window reader and the relayout run one thread each")
    dfbs = [
        E.DataflowBufferSpec(
            unique_id="input_window",
            entry_size=p.page_bytes,
            num_entries=p.input_window_depth * p.window_entries,
            data_format_metadata=RING_DATA_FORMAT,
        ),
        E.DataflowBufferSpec(
            unique_id="output_tiles",
            entry_size=p.page_bytes,
            num_entries=OUTPUT_RING_DEPTH * p.output_quantum,
            data_format_metadata=RING_DATA_FORMAT,
        ),
    ]
    defines = {"RESHAPE_TILE_ROW_REGROUP": "1"}
    common_ct = {
        **plan,
        "tensor_w_tiles": p.tensor_w_tiles,
        "block_tile_cols": p.block_tile_cols,
        "block_out_tile_rows": p.block_out_tile_rows,
        "window_entries": p.window_entries,
        "page_bytes": p.page_bytes,
    }
    relayout_ct = {
        **common_ct,
        "h_in": p.h_in,
        "h_out": p.h_out,
        "tile_rows": TILE_ROWS,
        "face_rows": FACE_ROWS,
        "face_row_bytes": p.page_bytes // FACE_ROWS_PER_TILE,
        "face_bytes": p.page_bytes // FACES_PER_TILE,
    }
    reader = _window_reader_spec(common_ct, p.max_row_blocks, plan)
    relayout = E.KernelSpec(
        unique_id="relayout",
        source=str(KERNEL_DIR / "reshape_compute.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, all_explicit=True),
        num_threads=plan["relayout_threads"],
        dfb_bindings=[E.consumer_of("input_window", "window"), E.producer_of("output_tiles", "tiles")],
        compile_time_args=relayout_ct,
        compiler_options=E.KernelSpec.CompilerOptions(defines=defines),
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["col_count", "row_count", "in_tile_row0", "row_in_tile0", "row_in_plane0", "out_row0"]
        ),
    )
    writer = _tile_writer_spec(common_ct, plan)
    spec = E.ProgramSpec(
        name="reshape_tile_row_regroup",
        kernels=[reader, relayout, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=[
            E.TensorParameter(unique_id="input", spec=input_tensor.spec),
            E.TensorParameter(unique_id="output", spec=output_tensor.spec),
        ],
        work_units=[
            E.WorkUnitSpec(name="main", kernels=["reader", "relayout", "writer"], target_nodes=_core_set(p.cores))
        ],
    )
    per_core = lambda d: {c: d[c] for c in p.cores}  # noqa: E731
    num_row_blocks = {c: -(-p.row_count[c] // p.block_out_tile_rows) for c in p.cores}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(
                kernel="reader",
                runtime_arg_values={
                    "col_start": per_core(p.col_start),
                    "col_count": per_core(p.col_count),
                    "num_row_blocks": num_row_blocks,
                },
                runtime_array_arg_values={"window_first": per_core(p.window_first), "window_rows": per_core(p.window_rows)},
            ),
            E.KernelRunArgs(
                kernel="relayout",
                runtime_arg_values={
                    "col_count": per_core(p.col_count),
                    "row_count": per_core(p.row_count),
                    "in_tile_row0": {c: p.cursor[c][0] for c in p.cores},
                    "row_in_tile0": {c: p.cursor[c][1] for c in p.cores},
                    "row_in_plane0": {c: p.cursor[c][2] for c in p.cores},
                    "out_row0": {c: p.cursor[c][3] for c in p.cores},
                },
            ),
            E.KernelRunArgs(
                kernel="writer",
                runtime_arg_values={
                    "col_start": per_core(p.col_start),
                    "col_count": per_core(p.col_count),
                    "row_start": per_core(p.row_start),
                    "row_count": per_core(p.row_count),
                },
            ),
        ]
    )
    return spec, run_args


# ---------------------------------------------------------------------------------------------------------------
# Regime W: w_regroup_rm
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class WRegroupRmPlan:
    page_in: int  # logical bytes of an input row (W_in x element size): the flat stream has no padding
    page_out: int
    entry_in: int  # window entry: the input page rounded up to the L1 alignment (stays inside the page's slot)
    entry_out: int
    in_pages: int
    out_pages: int
    row_groups: int
    block_out_rows: int
    n_in_cap: int
    window_entries: int
    input_window_depth: int
    output_quantum: int
    max_row_blocks: int
    max_blocks_per_core: int
    cores: list
    # per core
    row_start: dict = field(default_factory=dict)
    row_count: dict = field(default_factory=dict)
    in_off0: dict = field(default_factory=dict)  # byte offset of the core's first output byte in its first input page
    window_first: dict = field(default_factory=dict)  # core -> [per row block]
    window_rows: dict = field(default_factory=dict)  # core -> [per row block]

    def input_pages_read(self):
        """Input pages the readers fetch in total (a structural count: the minimum is in_pages)."""
        return sum(sum(self.window_rows[c]) for c in self.cores)


def _page_window(ra, rb, page_in, page_out):
    """(first input page, number of input pages) holding the bytes of output rows [ra, rb)."""
    first = ra * page_out // page_in
    return first, (rb * page_out - 1) // page_in - first + 1


def plan_w_regroup_rm(input_tensor, output_tensor, *, block_out_rows=None, row_groups=None):
    """The split and block extents of regime W. Output rows are independent: they are cut into balanced contiguous
    ranges over the grid, `row_groups <= in_pages` so that each boundary between two ranges re-reads at most one
    input page (reads < 2 x minimum, the cap of regime B). The block is the whole per-core range unless the window
    does not fit; the knobs override the defaults for tests and refinements."""
    page_in, page_out = input_tensor.buffer_page_size(), output_tensor.buffer_page_size()
    in_pages, out_pages = input_tensor.buffer_num_pages(), output_tensor.buffer_num_pages()
    if in_pages * page_in != out_pages * page_out:
        raise RuntimeError(f"reshape w_regroup_rm: {in_pages} x {page_in} B in, {out_pages} x {page_out} B out")
    align = ttnn.get_l1_alignment()
    entry_in, entry_out = _round_up(page_in, align), _round_up(page_out, align)

    grid = input_tensor.device().compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    if row_groups is None:
        row_groups = min(out_pages, in_pages, num_cores)
    row_groups = max(1, min(row_groups, out_pages, num_cores))
    row_ranges = _balanced_ranges(out_pages, row_groups)
    max_core_rows = max(c for _, c in row_ranges)
    budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES

    def n_in_cap_for(bh):
        return max(_page_window(ka, kb, page_in, page_out)[1] for rs, rc in row_ranges for ka, kb in _row_blocks(rs, rc, bh))

    def quantum(bh):
        return min(bh, max(1, BLOCK_BYTES_TARGET // entry_out))

    def fits(bh):
        n_in = n_in_cap_for(bh)
        depth = 2 if -(-max_core_rows // bh) > 1 else 1
        window = depth * n_in * entry_in
        output = OUTPUT_RING_DEPTH * quantum(bh) * entry_out
        return (
            depth * n_in <= UINT16_MAX  # ring counts are uint16_t
            and window <= DFB_RING_MAX_BYTES
            and output <= DFB_RING_MAX_BYTES
            and window + output <= budget
        )

    bh = block_out_rows if block_out_rows is not None else max_core_rows
    bh = max(1, min(bh, max_core_rows))
    if not fits(bh):  # fit search: the largest block below the per-core range whose window fits
        if not fits(1):
            raise NotImplementedError(f"reshape: no block of the W regroup fits L1 ({page_in} B -> {page_out} B rows)")
        bh = _largest_fitting(fits, 1, bh)

    max_row_blocks = -(-max_core_rows // bh)
    n_in_cap = n_in_cap_for(bh)
    p = WRegroupRmPlan(
        page_in=page_in,
        page_out=page_out,
        entry_in=entry_in,
        entry_out=entry_out,
        in_pages=in_pages,
        out_pages=out_pages,
        row_groups=row_groups,
        block_out_rows=bh,
        n_in_cap=n_in_cap,
        window_entries=n_in_cap,
        input_window_depth=2 if max_row_blocks > 1 else 1,
        output_quantum=quantum(bh),
        max_row_blocks=max_row_blocks,
        max_blocks_per_core=max_row_blocks,
        cores=_grid_cores(grid, row_groups),
    )
    for core, (rs, rc) in zip(p.cores, row_ranges):
        blocks = [_page_window(ka, kb, page_in, page_out) for ka, kb in _row_blocks(rs, rc, bh)]
        p.row_start[core], p.row_count[core] = rs, rc
        p.in_off0[core] = rs * page_out - blocks[0][0] * page_in
        p.window_first[core] = [f for f, _ in blocks] + [0] * (max_row_blocks - len(blocks))
        p.window_rows[core] = [n for _, n in blocks] + [0] * (max_row_blocks - len(blocks))
    return p


def _largest_fitting(fits, lo, hi):
    """The largest bh in [lo, hi) with fits(bh), given fits(lo) (bisection: n_in_cap grows with bh)."""
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


def _w_regroup_rm_artifacts(input_tensor, output_tensor, p: WRegroupRmPlan, plan):
    if plan["reader_threads"] != 1 or plan["relayout_threads"] != 1:
        raise ValueError("reshape w_regroup_rm: the window reader and the relayout run one thread each")
    dfbs = [
        E.DataflowBufferSpec(
            unique_id="input_window",
            entry_size=p.entry_in,
            num_entries=p.input_window_depth * p.window_entries,
            data_format_metadata=RING_DATA_FORMAT,
        ),
        E.DataflowBufferSpec(
            unique_id="output_tiles",
            entry_size=p.entry_out,
            num_entries=OUTPUT_RING_DEPTH * p.output_quantum,
            data_format_metadata=RING_DATA_FORMAT,
        ),
    ]
    # A row-major tensor is a one-column grid of pages for the window reader.
    reader_ct = {
        **plan,
        "tensor_w_tiles": 1,
        "block_tile_cols": 1,
        "window_entries": p.window_entries,
        "page_bytes": p.entry_in,
    }
    relayout_ct = {
        **plan,
        "block_out_rows": p.block_out_rows,
        "window_entries": p.window_entries,
        "page_in": p.page_in,
        "page_out": p.page_out,
        "entry_in": p.entry_in,
    }
    reader = _window_reader_spec(reader_ct, p.max_row_blocks, plan)
    relayout = E.KernelSpec(
        unique_id="relayout",
        source=str(KERNEL_DIR / "reshape_compute.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, all_explicit=True),
        num_threads=plan["relayout_threads"],
        dfb_bindings=[E.consumer_of("input_window", "window"), E.producer_of("output_tiles", "tiles")],
        compile_time_args=relayout_ct,
        compiler_options=E.KernelSpec.CompilerOptions(defines={"RESHAPE_W_REGROUP_RM": "1"}),
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["row_count", "in_off0"]),
    )
    # The output pages leave in order, one implicit-sync write per entry: regime A's page writer.
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "reshape_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config),
        num_threads=plan["writer_threads"],
        dfb_bindings=[E.consumer_of("output_tiles", "out_ring")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args={**plan, "block_pages": p.block_out_rows},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["core_pages", "start_page"]),
    )
    spec = E.ProgramSpec(
        name="reshape_w_regroup_rm",
        kernels=[reader, relayout, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=[
            E.TensorParameter(unique_id="input", spec=input_tensor.spec),
            E.TensorParameter(unique_id="output", spec=output_tensor.spec),
        ],
        work_units=[
            E.WorkUnitSpec(name="main", kernels=["reader", "relayout", "writer"], target_nodes=_core_set(p.cores))
        ],
    )
    per_core = lambda d: {c: d[c] for c in p.cores}  # noqa: E731
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(
                kernel="reader",
                runtime_arg_values={
                    "col_start": {c: 0 for c in p.cores},
                    "col_count": {c: 1 for c in p.cores},
                    "num_row_blocks": {c: -(-p.row_count[c] // p.block_out_rows) for c in p.cores},
                },
                runtime_array_arg_values={"window_first": per_core(p.window_first), "window_rows": per_core(p.window_rows)},
            ),
            E.KernelRunArgs(
                kernel="relayout",
                runtime_arg_values={"row_count": per_core(p.row_count), "in_off0": per_core(p.in_off0)},
            ),
            E.KernelRunArgs(
                kernel="writer",
                runtime_arg_values={"core_pages": per_core(p.row_count), "start_page": per_core(p.row_start)},
            ),
        ]
    )
    return spec, run_args


# ---------------------------------------------------------------------------------------------------------------
# Regime W (tile): w_regroup_tile
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class WRegroupTilePlan:
    page_bytes: int
    h_in: int
    w_in: int
    h_out: int
    w_out: int
    in_w_tiles: int
    out_w_tiles: int
    tensor_in_tile_rows: int
    tensor_out_tile_rows: int
    row_groups: int
    col_groups: int
    block_out_tile_rows: int
    window_cols: int  # input tile columns per window tile-row (the widest cluster's column window)
    out_cols: int  # output tile columns of the widest cluster
    n_in_cap: int
    window_entries: int
    input_window_depth: int
    output_quantum: int
    max_row_blocks: int
    max_blocks_per_core: int
    cores: list
    # per core
    row_start: dict = field(default_factory=dict)  # output tile-rows [row_start, row_start + row_count)
    row_count: dict = field(default_factory=dict)
    col_start: dict = field(default_factory=dict)  # output tile columns [col_start, col_start + col_count)
    col_count: dict = field(default_factory=dict)
    win_col0: dict = field(default_factory=dict)  # input tile columns [win_col0, win_col0 + win_cols) of the window
    win_cols: dict = field(default_factory=dict)
    window_first: dict = field(default_factory=dict)  # core -> [per row block]
    window_rows: dict = field(default_factory=dict)  # core -> [per row block]
    cursor: dict = field(default_factory=dict)  # core -> (in_tile_row, row_in_tile, row_in_plane, col_in, out_row0)

    def input_tiles_read(self):
        """Input tiles the readers fetch in total (a structural count: the minimum is in_tile_rows x in_w_tiles)."""
        return sum(sum(self.window_rows[c]) * self.win_cols[c] for c in self.cores)

    def minimum_input_tiles(self):
        return self.tensor_in_tile_rows * self.in_w_tiles


class _FlatGeometry(_RowGeometry):
    """Flat element index f <-> input (tile-row, row, column) and output tile-rows, when W changes too: output row g
    holds flat elements [g * w_out, (g + 1) * w_out), input row f // w_in holds f."""

    def __init__(self, in_shape, out_shape):
        (h_in, self.w_in), (h_out, self.w_out) = _hw(in_shape), _hw(out_shape)
        super().__init__(h_in, h_out)
        self.in_w_tiles = -(-self.w_in // TILE_ROWS)
        self.out_w_tiles = -(-self.w_out // TILE_ROWS)

    def out_rows(self, ka, kb):
        """Global output rows [g0, g1) of output tile-rows [ka, kb) (consecutive: padding rows are not rows)."""
        return self.first_global_row(ka), self.first_global_row(kb - 1) + self.valid_rows(kb - 1)

    def col_span(self, j0, jc):
        """Output element columns [c_lo, c_hi) of output tile columns [j0, j0 + jc)."""
        return TILE_ROWS * j0, min(TILE_ROWS * (j0 + jc), self.w_out)

    def window(self, ka, kb, c_lo, c_hi):
        """(first input tile-row, number of input tile-rows) holding columns [c_lo, c_hi) of output tile-rows [ka, kb):
        the block's first element has the smallest flat index, its last the largest."""
        g0, g1 = self.out_rows(ka, kb)
        first = self.in_tile_row_of((g0 * self.w_out + c_lo) // self.w_in)
        last = self.in_tile_row_of(((g1 - 1) * self.w_out + c_hi - 1) // self.w_in)
        return first, last - first + 1

    def col_window(self, g0, g1, c_lo, c_hi):
        """(first input tile column, count) bounding the input columns of output rows [g0, g1), columns [c_lo, c_hi).
        (g * w_out) mod w_in repeats every w_in / gcd(w_in, w_out) rows, so that many rows decide it; a span that
        crosses an input row end takes every column."""
        lo, hi = self.w_in, -1
        for g in range(g0, min(g1, g0 + self.w_in // math.gcd(self.w_in, self.w_out))):
            f0, f1 = g * self.w_out + c_lo, g * self.w_out + c_hi - 1
            if f0 // self.w_in != f1 // self.w_in:
                return 0, self.in_w_tiles
            lo, hi = min(lo, f0 % self.w_in), max(hi, f1 % self.w_in)
        return lo // TILE_ROWS, hi // TILE_ROWS - lo // TILE_ROWS + 1

    def cursor(self, g, c):
        """The relayout's input cursor at output row g, column c: (input tile-row, row in that tile, row in the input
        plane, input column), and the output row in its plane."""
        g_in, col = divmod(g * self.w_out + c, self.w_in)
        row_in_plane = g_in % self.h_in
        return (self.in_tile_row_of(g_in), row_in_plane % TILE_ROWS, row_in_plane, col, g % self.h_out)


def plan_w_regroup_tile(
    input_tensor, in_shape, out_shape, *, block_out_tile_rows=None, row_groups=None, col_groups=None
):
    """The split and block extents of regime W in tile layout. Output tile-rows are split like regime B's
    (`row_groups <= in_tile_rows`: one boundary input tile-row re-read per range boundary); output tile columns are
    split only as far as the input reads stay within max(2 x minimum, the column-unsplit reads), because the input
    columns of different output columns overlap when W changes. The block is the whole per-core range unless the
    window does not fit: the fit search shrinks the row block, then splits columns further. The keyword knobs
    override the defaults for tests and refinements."""
    page_bytes = input_tensor.buffer_page_size()
    geo = _FlatGeometry(in_shape, out_shape)
    volume = math.prod(in_shape)
    t_in = (volume // (geo.h_in * geo.w_in)) * geo.pin
    t_out = (volume // (geo.h_out * geo.w_out)) * geo.pout
    minimum = t_in * geo.in_w_tiles

    grid = input_tensor.device().compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    if row_groups is None:
        row_groups = min(t_out, t_in, num_cores)
    row_groups = max(1, min(row_groups, t_out, num_cores))
    row_ranges = _balanced_ranges(t_out, row_groups)
    max_core_rows = max(c for _, c in row_ranges)
    max_col_groups = max(1, min(geo.out_w_tiles, num_cores // row_groups))
    budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES
    quantum_cap = max(1, BLOCK_BYTES_TARGET // page_bytes)

    def split(cg):
        """Per core (rs, rc, j0, jc, win_col0, win_cols) for `cg` column groups."""
        out = []
        for rs, rc in row_ranges:
            g0, g1 = geo.out_rows(rs, rs + rc)
            for j0, jc in _balanced_ranges(geo.out_w_tiles, cg):
                out.append((rs, rc, j0, jc, *geo.col_window(g0, g1, *geo.col_span(j0, jc))))
        return out

    def windows(core, bh):
        rs, rc, j0, jc = core[:4]
        return [geo.window(ka, kb, *geo.col_span(j0, jc)) for ka, kb in _row_blocks(rs, rc, bh)]

    def reads(cores, bh):
        return sum(sum(n for _, n in windows(c, bh)) * c[5] for c in cores)

    def fits(cores, bh):
        n_in = max(n for c in cores for _, n in windows(c, bh))
        depth = 2 if -(-max_core_rows // bh) > 1 else 1
        entries = depth * n_in * max(c[5] for c in cores)
        output = OUTPUT_RING_DEPTH * min(max(c[3] for c in cores), quantum_cap) * page_bytes
        return (
            entries <= UINT16_MAX
            and entries * page_bytes <= DFB_RING_MAX_BYTES
            and output <= DFB_RING_MAX_BYTES
            and entries * page_bytes + output <= budget
        )

    if col_groups is None:
        bound = max(2 * minimum, reads(split(1), max_core_rows))
        col_groups = max(c for c in range(1, max_col_groups + 1) if reads(split(c), max_core_rows) <= bound)
    col_groups = max(1, min(col_groups, geo.out_w_tiles, num_cores // row_groups))

    # Fit search: the whole per-core range first, then the largest row block that fits (a single block needs no
    # second window, so it can fit where a smaller one does not), then split the columns further.
    bh0 = max(1, min(block_out_tile_rows if block_out_tile_rows is not None else max_core_rows, max_core_rows))

    def fitting_block(cores):
        if fits(cores, bh0):
            return bh0
        return _largest_fitting(lambda b: fits(cores, b), 1, bh0) if fits(cores, 1) else None

    cores = split(col_groups)
    bh = fitting_block(cores)
    while bh is None and col_groups < max_col_groups:
        col_groups += 1
        cores = split(col_groups)
        bh = fitting_block(cores)
    if bh is None:
        raise NotImplementedError(f"reshape: no block of the W regroup fits L1 for {in_shape} -> {out_shape}")

    max_row_blocks = -(-max_core_rows // bh)
    n_in_cap = max(n for c in cores for _, n in windows(c, bh))
    window_cols = max(c[5] for c in cores)
    out_cols = max(c[3] for c in cores)
    p = WRegroupTilePlan(
        page_bytes=page_bytes,
        h_in=geo.h_in,
        w_in=geo.w_in,
        h_out=geo.h_out,
        w_out=geo.w_out,
        in_w_tiles=geo.in_w_tiles,
        out_w_tiles=geo.out_w_tiles,
        tensor_in_tile_rows=t_in,
        tensor_out_tile_rows=t_out,
        row_groups=row_groups,
        col_groups=col_groups,
        block_out_tile_rows=bh,
        window_cols=window_cols,
        out_cols=out_cols,
        n_in_cap=n_in_cap,
        window_entries=n_in_cap * window_cols,
        input_window_depth=2 if max_row_blocks > 1 else 1,
        output_quantum=min(out_cols, quantum_cap),
        max_row_blocks=max_row_blocks,
        max_blocks_per_core=max_row_blocks,
        cores=_grid_cores(grid, len(cores)),
    )
    for core, c in zip(p.cores, cores):
        rs, rc, j0, jc, wc0, wcs = c
        blocks = windows(c, bh)
        p.row_start[core], p.row_count[core] = rs, rc
        p.col_start[core], p.col_count[core] = j0, jc
        p.win_col0[core], p.win_cols[core] = wc0, wcs
        p.window_first[core] = [f for f, _ in blocks] + [0] * (max_row_blocks - len(blocks))
        p.window_rows[core] = [n for _, n in blocks] + [0] * (max_row_blocks - len(blocks))
        p.cursor[core] = geo.cursor(geo.first_global_row(rs), geo.col_span(j0, jc)[0])
    return p


def _w_regroup_tile_artifacts(input_tensor, output_tensor, p: WRegroupTilePlan, plan):
    if plan["reader_threads"] != 1 or plan["relayout_threads"] != 1:
        raise ValueError("reshape w_regroup_tile: the window reader and the relayout run one thread each")
    dfbs = [
        E.DataflowBufferSpec(
            unique_id="input_window",
            entry_size=p.page_bytes,
            num_entries=p.input_window_depth * p.window_entries,
            data_format_metadata=RING_DATA_FORMAT,
        ),
        E.DataflowBufferSpec(
            unique_id="output_tiles",
            entry_size=p.page_bytes,
            num_entries=OUTPUT_RING_DEPTH * p.output_quantum,
            data_format_metadata=RING_DATA_FORMAT,
        ),
    ]
    # The window is the input tile-rows x the cluster's input column window, `window_cols` entries per tile-row.
    reader_ct = {
        **plan,
        "tensor_w_tiles": p.in_w_tiles,
        "block_tile_cols": p.window_cols,
        "window_entries": p.window_entries,
        "page_bytes": p.page_bytes,
    }
    face_row_bytes = p.page_bytes // FACE_ROWS_PER_TILE
    relayout_ct = {
        **plan,
        "block_out_tile_rows": p.block_out_tile_rows,
        "window_entries": p.window_entries,
        "window_cols": p.window_cols,
        "page_bytes": p.page_bytes,
        "h_in": p.h_in,
        "w_in": p.w_in,
        "h_out": p.h_out,
        "w_out": p.w_out,
        "tile_rows": TILE_ROWS,
        "face_rows": FACE_ROWS,
        "face_row_bytes": face_row_bytes,
        "face_bytes": p.page_bytes // FACES_PER_TILE,
        "elem_bytes": face_row_bytes // FACE_ROWS,  # a face row is FACE_ROWS elements
    }
    writer_ct = {
        **plan,
        "tensor_w_tiles": p.out_w_tiles,
        "block_tile_cols": p.out_cols,
        "block_out_tile_rows": p.block_out_tile_rows,
    }
    reader = _window_reader_spec(reader_ct, p.max_row_blocks, plan)
    relayout = E.KernelSpec(
        unique_id="relayout",
        source=str(KERNEL_DIR / "reshape_compute.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, all_explicit=True),
        num_threads=plan["relayout_threads"],
        dfb_bindings=[E.consumer_of("input_window", "window"), E.producer_of("output_tiles", "tiles")],
        compile_time_args=relayout_ct,
        compiler_options=E.KernelSpec.CompilerOptions(defines={"RESHAPE_W_REGROUP_TILE": "1"}),
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=[
                "col_count",
                "row_count",
                "win_col0",
                "last_tile_cols",
                "in_tile_row0",
                "row_in_tile0",
                "row_in_plane0",
                "col_in0",
                "out_row0",
            ]
        ),
    )
    writer = _tile_writer_spec(writer_ct, plan)
    spec = E.ProgramSpec(
        name="reshape_w_regroup_tile",
        kernels=[reader, relayout, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=[
            E.TensorParameter(unique_id="input", spec=input_tensor.spec),
            E.TensorParameter(unique_id="output", spec=output_tensor.spec),
        ],
        work_units=[
            E.WorkUnitSpec(name="main", kernels=["reader", "relayout", "writer"], target_nodes=_core_set(p.cores))
        ],
    )
    per_core = lambda d: {c: d[c] for c in p.cores}  # noqa: E731
    last_cols = {
        c: min(TILE_ROWS, p.w_out - TILE_ROWS * (p.col_start[c] + p.col_count[c] - 1)) for c in p.cores
    }
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(
                kernel="reader",
                runtime_arg_values={
                    "col_start": per_core(p.win_col0),
                    "col_count": per_core(p.win_cols),
                    "num_row_blocks": {c: -(-p.row_count[c] // p.block_out_tile_rows) for c in p.cores},
                },
                runtime_array_arg_values={"window_first": per_core(p.window_first), "window_rows": per_core(p.window_rows)},
            ),
            E.KernelRunArgs(
                kernel="relayout",
                runtime_arg_values={
                    "col_count": per_core(p.col_count),
                    "row_count": per_core(p.row_count),
                    "win_col0": per_core(p.win_col0),
                    "last_tile_cols": last_cols,
                    **{name: {c: p.cursor[c][i] for c in p.cores} for i, name in enumerate(
                        ["in_tile_row0", "row_in_tile0", "row_in_plane0", "col_in0", "out_row0"])},
                },
            ),
            E.KernelRunArgs(
                kernel="writer",
                runtime_arg_values={
                    "col_start": per_core(p.col_start),
                    "col_count": per_core(p.col_count),
                    "row_start": per_core(p.row_start),
                    "row_count": per_core(p.row_count),
                },
            ),
        ]
    )
    return spec, run_args


# ---------------------------------------------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------------------------------------------


def create_program_artifacts(input_tensor, output_tensor, in_shape, out_shape, *, plan=THREAD_PLAN, **knobs):
    """(regime, block plan, ProgramSpec, ProgramRunArgs, tensor indices) of one reshape call.

    `knobs` override block extents for tests and refinements: `block_bytes_target` (regime A),
    `block_tile_cols`, `block_out_tile_rows`, `row_groups` (regime B), `block_out_rows`, `row_groups` (regime W, row-major),
    `block_out_tile_rows`, `row_groups`, `col_groups` (regime W, tile)."""
    validate_thread_plan(plan)
    regime = select_regime(in_shape, out_shape, input_tensor.layout)
    if regime == REGIME_PAGE_IDENTITY:
        p = plan_page_identity(
            input_tensor, output_tensor, block_bytes_target=knobs.get("block_bytes_target", BLOCK_BYTES_TARGET), plan=plan
        )
        spec, run_args = _page_identity_artifacts(input_tensor, output_tensor, p, plan)
    elif regime == REGIME_W_REGROUP_RM:
        p = plan_w_regroup_rm(
            input_tensor, output_tensor, block_out_rows=knobs.get("block_out_rows"), row_groups=knobs.get("row_groups")
        )
        spec, run_args = _w_regroup_rm_artifacts(input_tensor, output_tensor, p, plan)
    elif regime == REGIME_W_REGROUP_TILE:
        p = plan_w_regroup_tile(
            input_tensor,
            in_shape,
            out_shape,
            block_out_tile_rows=knobs.get("block_out_tile_rows"),
            row_groups=knobs.get("row_groups"),
            col_groups=knobs.get("col_groups"),
        )
        spec, run_args = _w_regroup_tile_artifacts(input_tensor, output_tensor, p, plan)
    else:
        p = plan_tile_row_regroup(
            input_tensor,
            in_shape,
            out_shape,
            block_tile_cols=knobs.get("block_tile_cols"),
            block_out_tile_rows=knobs.get("block_out_tile_rows"),
            row_groups=knobs.get("row_groups"),
        )
        spec, run_args = _tile_row_regroup_artifacts(input_tensor, output_tensor, p, plan)
    return regime, p, spec, run_args, {"input": 0, "output": 1}
