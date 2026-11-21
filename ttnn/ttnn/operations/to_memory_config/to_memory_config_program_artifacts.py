# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for to_memory_config: the input's pages -> the output's pages, same bytes.

One reader kernel and one writer kernel per engaged cluster, sharing one staging ring (`dfb_pages`); no
compute kernel (the op computes nothing, op_design.md "Thread plan"). Both ends of the ring use implicit sync:
a read tagged NocOptions::TXN_ID lands at the ring's cursor and DM0's interrupt handler posts its credit, a
tagged write leaves from the ring's cursor and the handler acks it.

The unit the ring carries is an *entry* (op_design.md, "Entry space"). In TILE layout an entry is a tile, which
is a page on both sides. In ROW_MAJOR layout a page is a whole row when the tensor is interleaved or height-sharded
and a `shard_w`-wide run of one row when it is width-sharded (the accessor's page shape is (1, shard_w)), so the two
sides may cut a row differently: an entry is then a run of `L = lcm(Pi, Po)` elements of one row, `L / Pi` input
pages and `L / Po` output pages. Entry `g` holds input pages `g * in_pieces + i` and output pages
`g * out_pieces + j`, so no page straddles an entry. A side with one page per entry keeps implicit sync; a side that
assembles or splits an entry from several pieces uses explicit sync with every transfer tagged (target page, "Ring
protocols"). Quasar's NoC needs no address alignment (`NOC_L1/DRAM_*_ALIGNMENT_BYTES` = 1), so a piece may sit at
any byte offset of its entry.

Which cluster moves which entries is the regime (op_design.md, Regimes); every regime is the same kernel pair
with a different table of (core, first entry, entry count, run length, run stride):

    R-in   the input is sharded (and has at least as many shards as the output): shard k's entries are moved by
           the k-th core of the input's shard grid, so its reads are core-local
    R-out  the output is sharded: shard k's entries are pulled by the k-th core of the output's grid, so its
           writes are core-local
    R-bal  neither side can own its entries (both interleaved, or a width shard narrower than an entry): entries
           split evenly over the whole worker grid

A height shard owns one contiguous run of entries; a width shard owns the same `run_len` columns of every entry
row (`run_stride` = entries per row). Correctness does not depend on the regime: an entry is read and written
through the two accessors under the same global entry id, so a core table only decides traffic.

Knobs (single source each; tests monkeypatch them): THREAD_PLAN, RING_BYTES_CAP, BUFFER_DEPTH,
TXN_ENTRIES_CAP, L1_RESERVE_BYTES, PAGE_STRIDE. The ring's entry count and the kernels' block trip counts
derive from `block_pages`, computed once in `ring_geometry`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# --- thread plan (R readers, C compute threads, W writers) per cluster -------------------------------------
# Bound by data movement (one NoC read and one NoC write per page, zero compute): one hart of each kind with
# implicit sync, and no NEO (target page, "Choosing the plan"). The kernels take R and W as compile-time args
# and stride their page loops by them from get_my_thread_id().
THREAD_PLAN = (1, 0, 1)
FREE_DM_HARTS = 6   # of 8 per cluster: DM0 serves the ring interrupts, DM1 is the remapper (target page)

# --- ring sizing ------------------------------------------------------------------------------------------
RING_BYTES_CAP = 262144      # the most L1 the ring spends: it shares L1 with L1-resident tensors and shards
BUFFER_DEPTH = 2             # blocks in flight when a core's whole assignment does not fit one block
TXN_ENTRIES_CAP = 255        # implicit-sync entries per transaction id fit a uint8
L1_RESERVE_BYTES = 0         # unreserved L1 kept back from the ring budget
NOC_MAX_PACKET_BYTES = 65536 # one implicit STRIDED transfer is one NoC packet (NOC_MAX_BURST_SIZE)

# --- work distribution ------------------------------------------------------------------------------------
PAGE_STRIDE = 1              # page ids of a core are start_page + i * PAGE_STRIDE (a bank-owner split turns it)

DFB_PAGES = "dfb_pages"
TP_INPUT = "input"
TP_OUTPUT = "output"
K_READER = "reader"
K_WRITER = "writer"


def validate_thread_plan(plan):
    r, c, w = plan
    if not (r >= 1 and w >= 1 and c == 0 and r + w <= FREE_DM_HARTS):
        raise ValueError(f"thread plan (R, C, W) = {plan}: need R, W >= 1, R + W <= {FREE_DM_HARTS}, C = 0")


# ---------------------------------------------------------------------------
# Page space
# ---------------------------------------------------------------------------


def page_count(tensor) -> int:
    """Pages of the tensor's page space: tiles (each image tile-padded) in TILE, rows in ROW_MAJOR."""
    padded = [int(d) for d in tensor.padded_shape]
    if tensor.layout == ttnn.TILE_LAYOUT:
        tile_h, tile_w = tensor.tile.tile_shape
        return math.prod(padded[:-2]) * (padded[-2] // tile_h) * (padded[-1] // tile_w)
    return math.prod(padded[:-1])


def _row_page_width(tensor) -> int:
    """Elements of one row a ROW_MAJOR page holds: the shard width when width-sharded, else the whole row."""
    config = tensor.memory_config()
    if config.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return int(config.shard_spec.shape[1])
    return int(tensor.padded_shape[-1])


@dataclass
class EntrySpace:
    in_pieces: int        # input pages per entry
    out_pieces: int       # output pages per entry
    in_piece_bytes: int   # logical bytes of one input page (one sized transfer when in_pieces > 1)
    out_piece_bytes: int  # logical bytes of one output page
    entry_cols: int       # entries per entry row (tile columns in TILE, W / L in ROW_MAJOR)
    entry_width: int      # elements of a row one entry spans (TILE: tile width)
    num_entries: int

    @property
    def entry_rows(self) -> int:
        return self.num_entries // self.entry_cols

    @property
    def entry_logical_bytes(self) -> int:
        return self.in_pieces * self.in_piece_bytes


def entry_space(input_tensor, output_tensor) -> EntrySpace:
    """The op's single index space (module docstring). Input and output have one shape, dtype and layout."""
    padded = [int(d) for d in input_tensor.padded_shape]
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        tile_w = input_tensor.tile.tile_shape[1]
        page = input_tensor.buffer_page_size()
        return EntrySpace(1, 1, page, page, padded[-1] // tile_w, tile_w, page_count(input_tensor))
    width = padded[-1]
    p_in, p_out = _row_page_width(input_tensor), _row_page_width(output_tensor)
    for p in (p_in, p_out):
        if width % p:
            # A partial last width shard has a shorter last page in every row: entries would straddle pages.
            raise NotImplementedError(f"to_memory_config: a row-major width shard of {p} columns does not divide W={width}")
    elem = input_tensor.buffer_page_size() // p_in
    run = math.lcm(p_in, p_out)
    rows = math.prod(padded[:-1])
    return EntrySpace(run // p_in, run // p_out, p_in * elem, p_out * elem, width // run, run, rows * (width // run))


@dataclass
class ShardGrid:
    cores: list          # cores in shard order
    kind: str            # "height" | "width"
    shard_entries: int   # height: entries per shard (a contiguous run); width: entry columns per shard


def _shard_grid(tensor, space: EntrySpace):
    """The tensor's shard grid in entries, or None when it is interleaved or its shards cannot own whole entries
    (a row-major width shard narrower than an entry)."""
    config = tensor.memory_config()
    kind = {ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height", ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width"}.get(
        config.memory_layout)
    if kind is None:
        return None
    shard = config.shard_spec
    row_wise = shard.orientation == ttnn.ShardOrientation.ROW_MAJOR
    cores = list(ttnn.corerange_to_cores(shard.grid, None, row_wise))
    shard_h, shard_w = (int(d) for d in shard.shape)
    if shard_w % space.entry_width:
        return None
    shard_cols = shard_w // space.entry_width
    if kind == "width":
        return ShardGrid(cores, kind, shard_cols)
    shard_rows = shard_h // tensor.tile.tile_shape[0] if tensor.layout == ttnn.TILE_LAYOUT else shard_h
    return ShardGrid(cores, kind, shard_rows * shard_cols)


# ---------------------------------------------------------------------------
# Work distribution: the regimes
# ---------------------------------------------------------------------------


@dataclass
class WorkSplit:
    regime: str          # "R-in" | "R-out" | "R-bal"
    cores: list          # engaged cores, each with at least one entry
    starts: dict         # core -> first entry id
    counts: dict         # core -> entry count
    run_lens: dict = None     # core -> entries per run (a width shard's columns); None: one run per core
    run_strides: dict = None  # core -> entry-id step between runs (entries per entry row)

    def __post_init__(self):
        if self.run_lens is None:
            self.run_lens = dict(self.counts)
        if self.run_strides is None:
            self.run_strides = {core: 0 for core in self.cores}


def _shards_with_data(grid: ShardGrid, space: EntrySpace) -> int:
    total = space.entry_cols if grid.kind == "width" else space.num_entries
    return -(-total // grid.shard_entries)


def _owner_split(regime, grid: ShardGrid, space: EntrySpace) -> WorkSplit:
    """Shard k's entries go to the k-th core of the shard grid; cores past the data get no work."""
    engaged, starts, counts, run_lens, run_strides = [], {}, {}, {}, {}
    for k, core in enumerate(grid.cores):
        first = k * grid.shard_entries
        if grid.kind == "height":
            n = min(grid.shard_entries, space.num_entries - first)
            run, total = n, n
        else:  # width: the same columns of every entry row
            run = min(grid.shard_entries, space.entry_cols - first)
            total = run * space.entry_rows
        if run <= 0:
            break
        engaged.append(core)
        starts[core], counts[core], run_lens[core], run_strides[core] = first, total, run, space.entry_cols
    return WorkSplit(regime, engaged, starts, counts, run_lens, run_strides)


def _balanced_split(num_entries, grid) -> WorkSplit:
    """Contiguous entry ranges over the whole worker grid, as split_work_to_cores deals them."""
    _, _, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, num_entries, row_wise=True)
    engaged, starts, counts, start = [], {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            engaged.append(core)
            starts[core], counts[core] = start, per_core
            start += per_core
    assert start == num_entries, (start, num_entries)
    return WorkSplit("R-bal", engaged, starts, counts)


def split_work(input_tensor, output_tensor, space: EntrySpace = None) -> WorkSplit:
    """The regime selection function of op_design.md."""
    space = space or entry_space(input_tensor, output_tensor)
    in_grid, out_grid = _shard_grid(input_tensor, space), _shard_grid(output_tensor, space)
    if in_grid is not None and (out_grid is None or _shards_with_data(in_grid, space) >= _shards_with_data(out_grid, space)):
        return _owner_split("R-in", in_grid, space)
    if out_grid is not None:
        return _owner_split("R-out", out_grid, space)
    return _balanced_split(space.num_entries, input_tensor.device().compute_with_storage_grid_size())


# ---------------------------------------------------------------------------
# Ring geometry: the block knob
# ---------------------------------------------------------------------------


@dataclass
class RingGeometry:
    entry_bytes: int     # bytes per ring entry = bytes per implicit-sync transfer
    block_pages: int     # pages per block (the page-axis block knob)
    depth: int           # blocks the ring holds
    num_entries: int     # block_pages * depth


def ring_geometry(input_tensor, output_tensor, core_pages_max, space: EntrySpace = None) -> RingGeometry:
    """Coarsest block that fits: the whole per-core assignment when it fits one block, else BUFFER_DEPTH
    blocks of the largest size the ring budget and the per-txn-id entry cap allow. `core_pages_max` counts
    entries (a page when both sides have one page per entry)."""
    space = space or entry_space(input_tensor, output_tensor)
    if space.in_pieces == 1 and space.out_pieces == 1:
        # An implicit transfer moves exactly entry_size bytes, so an entry may not exceed either side's page slot
        # (a 128 B DRAM row slot into a 112 B L1 slot would trample the next row). Both slots hold the page.
        entry_bytes = min(input_tensor.buffer_aligned_page_size(), output_tensor.buffer_aligned_page_size())
    else:
        # A row run of several pieces on one side: the side with one page per entry moves entry_bytes from or to
        # that page's slot, which holds L * elem rounded up to its own alignment (>= the L1 alignment).
        align = ttnn.get_l1_alignment()
        entry_bytes = -(-space.entry_logical_bytes // align) * align
    if entry_bytes > NOC_MAX_PACKET_BYTES:
        raise NotImplementedError(f"to_memory_config: a {entry_bytes} B entry exceeds one NoC packet")
    ring_budget = min(RING_BYTES_CAP, ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES)
    single_block_cap = min(TXN_ENTRIES_CAP, ring_budget // entry_bytes)
    if core_pages_max <= single_block_cap:
        block_pages, depth = core_pages_max, 1
    else:
        block_pages, depth = min(TXN_ENTRIES_CAP, ring_budget // (BUFFER_DEPTH * entry_bytes)), BUFFER_DEPTH
    if block_pages < 1:
        raise NotImplementedError(f"to_memory_config: {BUFFER_DEPTH} pages of {entry_bytes} B exceed the ring budget")
    return RingGeometry(entry_bytes, block_pages, depth, block_pages * depth)


# ---------------------------------------------------------------------------
# The program
# ---------------------------------------------------------------------------


def _dm_config(make, explicit):
    """A side that assembles (reader) or splits (writer) an entry from several pieces keeps the ring's credits
    itself (reserve / push, wait / pop), so it opts the ring out of implicit sync (target page, "Ring protocols")."""
    hw = make()
    if explicit:
        hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=[DFB_PAGES])
    return hw


def create_program_artifacts(input_tensor, output_tensor, *, thread_plan=None):
    """(ProgramSpec, ProgramRunArgs, tensor indices) for ttnn.generic_op([input, output], ...)."""
    plan = THREAD_PLAN if thread_plan is None else thread_plan
    validate_thread_plan(plan)
    R, _, W = plan

    space = entry_space(input_tensor, output_tensor)
    work = split_work(input_tensor, output_tensor, space)
    ring = ring_geometry(input_tensor, output_tensor, max(work.counts.values()), space)

    dfb = E.DataflowBufferSpec(
        unique_id=DFB_PAGES, entry_size=ring.entry_bytes, num_entries=ring.num_entries,
        data_format_metadata=input_tensor.dtype)
    tensor_parameters = [
        E.TensorParameter(unique_id=TP_INPUT, spec=input_tensor.spec),
        E.TensorParameter(unique_id=TP_OUTPUT, spec=output_tensor.spec),
    ]
    compile_args = {
        "reader_threads": R, "writer_threads": W, "block_pages": ring.block_pages,
        "in_pieces": space.in_pieces, "in_piece_bytes": space.in_piece_bytes,
        "out_pieces": space.out_pieces, "out_piece_bytes": space.out_piece_bytes,
    }
    schema = E.KernelSpec.RuntimeArgSchema(
        runtime_arg_names=["first_entry", "entry_count", "entry_stride", "run_len", "run_stride"])

    reader = E.KernelSpec(
        unique_id=K_READER,
        source=str(KERNEL_DIR / "to_memory_config_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, explicit=space.in_pieces > 1),
        num_threads=R,
        dfb_bindings=[E.producer_of(DFB_PAGES, DFB_PAGES)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=TP_INPUT, accessor_name="input")],
        compile_time_args=compile_args,
        runtime_arg_schema=schema,
    )
    writer = E.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "to_memory_config_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config, explicit=space.out_pieces > 1),
        num_threads=W,
        dfb_bindings=[E.consumer_of(DFB_PAGES, DFB_PAGES)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=TP_OUTPUT, accessor_name="output")],
        compile_time_args=compile_args,
        runtime_arg_schema=schema,
    )
    target_nodes = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in work.cores])
    spec = E.ProgramSpec(
        name="to_memory_config",
        kernels=[reader, writer],
        dataflow_buffers=[dfb],
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=[K_READER, K_WRITER], target_nodes=target_nodes)],
    )
    dm_args = {
        "first_entry": work.starts,
        "entry_count": work.counts,
        "entry_stride": {core: PAGE_STRIDE for core in work.cores},
        "run_len": work.run_lens,
        "run_stride": work.run_strides,
    }
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel=K_READER, runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel=K_WRITER, runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {TP_INPUT: 0, TP_OUTPUT: 1}
