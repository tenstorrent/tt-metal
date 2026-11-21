# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts of reshape (`ttnn.ProgramSpec`, one `ttnn.generic_op` per call).

Two regimes (op_design.md, Blocking Model -> Regimes); each is its own ProgramSpec, selected on the host:

page_identity      output page p is input page p (row-major with W kept; tile layout when the tiles hold the same
                   rows before and after). Two DM kernels on every engaged cluster: the reader moves a contiguous
                   page range into `dfb_pages`, the writer moves it out. Both ring sides use implicit sync.

tile_row_regroup   tile layout, H changes and not both H's are multiples of 32: rows move between tiles. Three DM
                   kernels: the reader assembles a window of whole input tiles (explicit sync), the relayout
                   (`reshape_compute.cpp`, a DM kernel: the op computes nothing) copies face-row runs from the
                   window into each output tile, the writer sends the output tiles (implicit sync).

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

# Tile geometry: 32 x 32, four 16 x 16 faces (validated against the tensor's tile on the host). The single source:
# the relayout kernel receives `tile_rows` / `face_rows` as compile-time arguments.
TILE_ROWS = 32
FACE_ROWS = TILE_ROWS // 2
FACES_PER_TILE = 4
FACE_ROWS_PER_TILE = FACES_PER_TILE * FACE_ROWS

REGIME_PAGE_IDENTITY = "page_identity"
REGIME_TILE_ROW_REGROUP = "tile_row_regroup"


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
    """The regime-selection function of op_design.md (PHASE0 has already refused a change of W)."""
    h_in, _ = _hw(in_shape)
    h_out, _ = _hw(out_shape)
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
        unique_id="pages", entry_size=p.entry_bytes, num_entries=p.ring_entries, data_format_metadata=input_tensor.dtype
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
    dtype = input_tensor.dtype
    dfbs = [
        E.DataflowBufferSpec(
            unique_id="input_window",
            entry_size=p.page_bytes,
            num_entries=p.input_window_depth * p.window_entries,
            data_format_metadata=dtype,
        ),
        E.DataflowBufferSpec(
            unique_id="output_tiles",
            entry_size=p.page_bytes,
            num_entries=OUTPUT_RING_DEPTH * p.output_quantum,
            data_format_metadata=dtype,
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
    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "reshape_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, ["input_window"]),
        num_threads=plan["reader_threads"],
        dfb_bindings=[E.producer_of("input_window", "in_ring")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args=common_ct,
        compiler_options=E.KernelSpec.CompilerOptions(defines=defines),
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["col_start", "col_count", "num_row_blocks"],
            runtime_array_args=[
                E.ArrayArgDecl(name="window_first", size=p.max_row_blocks),
                E.ArrayArgDecl(name="window_rows", size=p.max_row_blocks),
            ],
        ),
    )
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
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "reshape_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config),
        num_threads=plan["writer_threads"],
        dfb_bindings=[E.consumer_of("output_tiles", "out_ring")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args=common_ct,
        compiler_options=E.KernelSpec.CompilerOptions(defines=defines),
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["col_start", "col_count", "row_start", "row_count"]
        ),
    )
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
# Entry
# ---------------------------------------------------------------------------------------------------------------


def create_program_artifacts(input_tensor, output_tensor, in_shape, out_shape, *, plan=THREAD_PLAN, **knobs):
    """(regime, block plan, ProgramSpec, ProgramRunArgs, tensor indices) of one reshape call.

    `knobs` override block extents for tests and refinements: `block_bytes_target` (regime A),
    `block_tile_cols`, `block_out_tile_rows`, `row_groups` (regime B)."""
    validate_thread_plan(plan)
    regime = select_regime(in_shape, out_shape, input_tensor.layout)
    if regime == REGIME_PAGE_IDENTITY:
        p = plan_page_identity(
            input_tensor, output_tensor, block_bytes_target=knobs.get("block_bytes_target", BLOCK_BYTES_TARGET), plan=plan
        )
        spec, run_args = _page_identity_artifacts(input_tensor, output_tensor, p, plan)
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
