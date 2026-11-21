# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for to_memory_config: page `p` of the input -> page `p` of the output.

One reader kernel and one writer kernel per engaged cluster, sharing one staging ring (`dfb_pages`); no
compute kernel (the op computes nothing, op_design.md "Thread plan"). Both ends of the ring use implicit sync:
a read tagged NocOptions::TXN_ID lands at the ring's cursor and DM0's interrupt handler posts its credit, a
tagged write leaves from the ring's cursor and the handler acks it.

Which cluster moves which pages is the regime (op_design.md, Regimes); every regime is the same kernel pair
with a different (core, start_page, num_pages) table:

    R-in   the input is height-sharded (and has at least as many shards as the output): shard k's pages are
           moved by the k-th core of the input's shard grid, so its reads are core-local
    R-out  the output is height-sharded: shard k's pages are pulled by the k-th core of the output's grid,
           so its writes are core-local
    R-bal  both sides interleaved: pages split evenly over the whole worker grid

Correctness does not depend on the regime: a page is read and written through the two accessors under the
same global page id, so a core table only decides traffic.

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


def _height_shard_grid(tensor):
    """(cores in shard order, pages per shard) of a height-sharded tensor, or None when it is interleaved."""
    config = tensor.memory_config()
    if config.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        return None
    shard = config.shard_spec
    row_wise = shard.orientation == ttnn.ShardOrientation.ROW_MAJOR
    cores = list(ttnn.corerange_to_cores(shard.grid, None, row_wise))
    shard_h, shard_w = (int(d) for d in shard.shape)
    if tensor.layout == ttnn.TILE_LAYOUT:
        tile_h, tile_w = tensor.tile.tile_shape
        pages_per_shard = (shard_h // tile_h) * (shard_w // tile_w)
    else:
        pages_per_shard = shard_h
    return cores, pages_per_shard


# ---------------------------------------------------------------------------
# Work distribution: the regimes
# ---------------------------------------------------------------------------


@dataclass
class WorkSplit:
    regime: str          # "R-in" | "R-out" | "R-bal"
    cores: list          # engaged cores, each with at least one page
    starts: dict         # core -> first page id
    counts: dict         # core -> page count


def _owner_split(regime, cores, pages_per_shard, num_pages) -> WorkSplit:
    """Shard k's page range goes to the k-th core of the shard grid; cores past the data get no work."""
    engaged, starts, counts = [], {}, {}
    for k, core in enumerate(cores):
        first = k * pages_per_shard
        n = min(pages_per_shard, num_pages - first)
        if n <= 0:
            break
        engaged.append(core)
        starts[core], counts[core] = first, n
    return WorkSplit(regime, engaged, starts, counts)


def _balanced_split(num_pages, grid) -> WorkSplit:
    """Contiguous page ranges over the whole worker grid, as split_work_to_cores deals them."""
    _, _, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, num_pages, row_wise=True)
    engaged, starts, counts, start = [], {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            engaged.append(core)
            starts[core], counts[core] = start, per_core
            start += per_core
    assert start == num_pages, (start, num_pages)
    return WorkSplit("R-bal", engaged, starts, counts)


def split_work(input_tensor, output_tensor) -> WorkSplit:
    """The regime selection function of op_design.md."""
    num_pages = page_count(input_tensor)
    in_grid, out_grid = _height_shard_grid(input_tensor), _height_shard_grid(output_tensor)
    shards = lambda g: -(-num_pages // g[1])  # noqa: E731  (shards that hold data)
    if in_grid is not None and (out_grid is None or shards(in_grid) >= shards(out_grid)):
        return _owner_split("R-in", *in_grid, num_pages)
    if out_grid is not None:
        return _owner_split("R-out", *out_grid, num_pages)
    return _balanced_split(num_pages, input_tensor.device().compute_with_storage_grid_size())


# ---------------------------------------------------------------------------
# Ring geometry: the block knob
# ---------------------------------------------------------------------------


@dataclass
class RingGeometry:
    entry_bytes: int     # bytes per ring entry = bytes per implicit-sync transfer
    block_pages: int     # pages per block (the page-axis block knob)
    depth: int           # blocks the ring holds
    num_entries: int     # block_pages * depth


def ring_geometry(input_tensor, output_tensor, core_pages_max) -> RingGeometry:
    """Coarsest block that fits: the whole per-core assignment when it fits one block, else BUFFER_DEPTH
    blocks of the largest size the ring budget and the per-txn-id entry cap allow."""
    # An implicit transfer moves exactly entry_size bytes, so an entry may not exceed either side's page slot
    # (a 128 B DRAM row slot into a 112 B L1 slot would trample the next row). Both slots hold the page.
    entry_bytes = min(input_tensor.buffer_aligned_page_size(), output_tensor.buffer_aligned_page_size())
    if entry_bytes > NOC_MAX_PACKET_BYTES:
        raise NotImplementedError(f"to_memory_config: a {entry_bytes} B page exceeds one NoC packet")
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


def create_program_artifacts(input_tensor, output_tensor, *, thread_plan=None):
    """(ProgramSpec, ProgramRunArgs, tensor indices) for ttnn.generic_op([input, output], ...)."""
    plan = THREAD_PLAN if thread_plan is None else thread_plan
    validate_thread_plan(plan)
    R, _, W = plan

    work = split_work(input_tensor, output_tensor)
    ring = ring_geometry(input_tensor, output_tensor, max(work.counts.values()))

    dfb = E.DataflowBufferSpec(
        unique_id=DFB_PAGES, entry_size=ring.entry_bytes, num_entries=ring.num_entries,
        data_format_metadata=input_tensor.dtype)
    tensor_parameters = [
        E.TensorParameter(unique_id=TP_INPUT, spec=input_tensor.spec),
        E.TensorParameter(unique_id=TP_OUTPUT, spec=output_tensor.spec),
    ]
    compile_args = {"reader_threads": R, "writer_threads": W, "block_pages": ring.block_pages}
    schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["start_page", "num_pages", "page_stride"])

    reader = E.KernelSpec(
        unique_id=K_READER,
        source=str(KERNEL_DIR / "to_memory_config_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=R,
        dfb_bindings=[E.producer_of(DFB_PAGES, DFB_PAGES)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=TP_INPUT, accessor_name="input")],
        compile_time_args=compile_args,
        runtime_arg_schema=schema,
    )
    writer = E.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "to_memory_config_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
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
        "start_page": work.starts,
        "num_pages": work.counts,
        "page_stride": {core: PAGE_STRIDE for core in work.cores},
    }
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel=K_READER, runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel=K_WRITER, runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {TP_INPUT: 0, TP_OUTPUT: 1}
