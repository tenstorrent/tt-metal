# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data-movement example: placement × NoC × operation.

Interleaved DRAM data movement across a LINE of `num_cores` Tensix cores, with three
switchable knobs so you can measure any cell of the matrix:

    op        : "read"  (DRAM->L1 only), "write" (L1->DRAM only), "copy" (identity read+write)
    noc       : "noc0" or "noc1"  -- the stream's NoC. For "copy", this is the READ NoC and
                                     writes take the other NoC (reads/writes never share links).
    placement : "column" (0,0)..(0,N-1)  -- the split_work_to_cores row_wise=False default / trap
                "row"    (0,0)..(N-1,0)
                "diagonal" (i,i)
                "grid"   the first N Tensix cores of the full worker grid, row-major (default: all of them)

One more knob applies to "copy" only:

    write_split : 0 = every output page is written on the write NoC (the other NoC from the reads).
                  k >= 1 = every k-th output page is written on the READ NoC instead, so 1/k of the
                  write stream moves to the other NoC (k=2: half, k=4: a quarter, k=1: all writes on
                  the read NoC). Reads never change. Issuing on both NoCs from one RISC-V needs
                  NOC_MODE::DM_DYNAMIC_NOC; noc_mode="dynamic" with write_split=0 is the matching
                  control (same mode, no split) so the mode's own cost is measured separately.

    swap_columns : 0 = every Tensix core reads on `noc` and writes on the other NoC (the usual pairing).
                   N >= 1 = the N worker-grid columns nearest a DRAM column swap the pairing (reads on the
                   other NoC, writes on `noc`); every other column keeps it. "all" = every column swaps.
                   Which columns are nearest DRAM is queried from the device (the logical worker columns the
                   device names as closest to each DRAM bank), never hard-coded. swap_anchor picks the
                   reference: "first" = only the DRAM column next to bank 0 (columns ranked by distance from
                   it), "every" = every DRAM column (ranked by distance from the nearest one). Each core still
                   uses one NoC per kernel (NOC_MODE dedicated); the swapped cores just get the other config.

The kernels are byte-identical across placements; only the physical cores and the chosen
NoC(s) change, so any measured difference is pure NoC routing. `read`/`write` isolate a
single stream (no CB handshake, no partner kernel) so the NoC-for-reads vs NoC-for-writes
question is answered without cross-contention. See README.md for the mechanism and
noc_placement_matrix.html (regenerate with noc_report.py) for the measured matrix.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
CB_ID = 0

# Switch space. Column is first so the "trap" is the baseline (it is what
# split_work_to_cores(..., row_wise=False), the default, hands you).
PLACEMENTS = ("column", "row", "diagonal")
# "grid" (the full worker grid, row-major) is accepted everywhere but kept out of PLACEMENTS so the
# line-placement matrices above stay as they are.
GRID = "grid"
ALL_PLACEMENTS = PLACEMENTS + (GRID,)
OPS = ("read", "write", "copy")
NOCS = ("noc0", "noc1")
NOC_MODES = ("dedicated", "dynamic")
SWAP_ANCHORS = ("first", "every")

_NOC = {"noc0": ttnn.NOC.NOC_0, "noc1": ttnn.NOC.NOC_1}
_OTHER = {"noc0": "noc1", "noc1": "noc0"}
_NOC_MODE = {"dedicated": ttnn.NOC_MODE.DM_DEDICATED_NOC, "dynamic": ttnn.NOC_MODE.DM_DYNAMIC_NOC}


def validate(input_tensor):
    """Keep the example dead simple: 2D, TILE, bf16, tile-aligned, interleaved."""
    shape = list(input_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"noc_placement example: rank must be 2, got {len(shape)}")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("noc_placement example: input must be TILE_LAYOUT")
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("noc_placement example: input must be bfloat16")
    h, w = shape
    if h % TILE or w % TILE:
        raise ValueError(f"noc_placement example: H and W must be multiples of {TILE}, got {shape}")


def _ordered_cores(placement, n, grid=None):
    """The N cores of a placement, in a fixed order so page assignment is identical
    across placements (core index k always gets the same page range; only its
    physical (x, y) differs)."""
    if placement == GRID:
        return [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]
    if placement == "row":
        return [ttnn.CoreCoord(i, 0) for i in range(n)]
    if placement == "column":
        return [ttnn.CoreCoord(0, i) for i in range(n)]
    if placement == "diagonal":
        return [ttnn.CoreCoord(i, i) for i in range(n)]
    raise ValueError(f"noc_placement example: placement must be one of {ALL_PLACEMENTS}, got {placement!r}")


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _assign_pages(num_pages, n):
    """Contiguous page ranges by core index; remainder spread over the first cores.
    Identical for every placement -> identical per-core work."""
    base, rem = divmod(num_pages, n)
    ranges = []
    start = 0
    for k in range(n):
        count = base + (1 if k < rem else 0)
        ranges.append((start, count))
        start += count
    return ranges


def _resolve_num_cores(device, num_cores, placement):
    """Cap a line at min(grid.x, grid.y) so a diagonal of this length fits on the grid;
    "grid" may use every Tensix core of the worker grid."""
    grid = device.compute_with_storage_grid_size()
    max_line = grid.x * grid.y if placement == GRID else min(grid.x, grid.y)
    if num_cores is None:
        return max_line
    if num_cores < 1 or num_cores > max_line:
        raise ValueError(
            f"noc_placement example: num_cores must be in [1, {max_line}] for placement {placement!r} "
            f"(a line must fit on the {grid.x}x{grid.y} grid as a diagonal), got {num_cores}"
        )
    return num_cores


def dram_adjacent_columns(device, anchor="first"):
    """Logical worker-grid columns ordered nearest-DRAM first (the order swap_columns consumes).

    Queried from the device, not hard-coded: the device names, per DRAM bank, the logical worker core
    closest to that bank; the columns of those cores are the DRAM-adjacent columns ("anchors").
    anchor="first": rank by distance from the column next to bank 0 only; "every": by distance from
    the nearest anchor column. Ties go to the lower column index."""
    if anchor not in SWAP_ANCHORS:
        raise ValueError(f"noc_placement example: swap_anchor must be one of {SWAP_ANCHORS}, got {anchor!r}")
    grid = device.compute_with_storage_grid_size()
    nearest = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
    anchors = [nearest[0].x] if anchor == "first" else sorted({c.x for c in nearest})
    return sorted(range(grid.x), key=lambda col: (min(abs(col - a) for a in anchors), col))


def _swapped_columns(device, swap_columns, anchor):
    """The set of logical worker columns whose Tensix cores run the swapped NoC pairing."""
    order = dram_adjacent_columns(device, anchor)
    n = len(order) if swap_columns == "all" else swap_columns
    if not 0 <= n <= len(order):
        raise ValueError(f"noc_placement example: swap_columns must be in [0, {len(order)}] or 'all', got {n}")
    return set(order[:n])


def _dm_config(processor, noc, noc_mode="dedicated"):
    return ttnn.DataMovementConfigDescriptor(processor=processor, noc=_NOC[noc], noc_mode=_NOC_MODE[noc_mode])


def _line(device, placement, num_cores, min_pages):
    """Resolve the placement into (cores, core_ranges, assignment, page_bytes)."""
    num_cores = _resolve_num_cores(device, num_cores, placement)
    if min_pages < num_cores:
        raise ValueError(f"noc_placement example: need >= {num_cores} pages, got {min_pages}")
    cores = _ordered_cores(placement, num_cores, device.compute_with_storage_grid_size())
    return cores, _core_range_set(cores), _assign_pages(min_pages, num_cores)


def _bench(input_tensor, *, op, noc, placement, num_cores, kernel_iters, block):
    """Single-stream read-only or write-only bench: one kernel, a fixed L1 scratch of
    `block` pages, no CB handshake and no partner kernel -- so only `op`'s NoC traffic
    is measured. `op` selects read_bench.cpp (RISCV_1) or write_bench.cpp (RISCV_0)."""
    device = input_tensor.device()
    page_bytes = input_tensor.buffer_aligned_page_size()
    num_pages = input_tensor.buffer_num_pages()
    cores, core_ranges, assignment = _line(device, placement, num_cores, num_pages)

    # A DRAM target the write bench writes into (and a placeholder the read bench ignores):
    # generic_op requires >= 1 input and >= 1 output tensor.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)), input_tensor.dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    target = output_tensor if op == "write" else input_tensor  # the tensor the kernel touches

    cb = ttnn.CBDescriptor(
        total_size=block * page_bytes,  # single fixed scratch region (bounded, page-count-independent)
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_ID, data_format=input_tensor.dtype, page_size=page_bytes)
        ],
    )

    ct_args = [page_bytes, kernel_iters, block]
    ct_args.extend(ttnn.TensorAccessorArgs(target).get_compile_time_args())

    rt_args = ttnn.RuntimeArgs()
    addr = target.buffer_address()
    for core, (start_page, count) in zip(cores, assignment):
        rt_args[core.x][core.y] = [addr, start_page, count]

    src = "read_bench.cpp" if op == "read" else "write_bench.cpp"
    proc = ttnn.DataMovementProcessor.RISCV_1 if op == "read" else ttnn.DataMovementProcessor.RISCV_0
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / src),
        core_ranges=core_ranges,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        config=_dm_config(proc, noc),  # the only thing that varies between NoC runs
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])
    return ttnn.generic_op([input_tensor, output_tensor], program)


def _copy(
    input_tensor,
    *,
    noc,
    placement,
    num_cores,
    kernel_iters,
    block,
    memory_config,
    write_split,
    noc_mode,
    swap_columns=0,
    swap_anchor="first",
):
    """Identity DRAM->DRAM copy: reader on `noc`, writer on the other NoC (so the read
    and write streams never share links). Output equals input for every placement/NoC.
    write_split=k moves every k-th output page's write onto the read NoC (see module doc).
    swap_columns=N gives the Tensix cores in the N DRAM-nearest columns the reversed pairing."""
    device = input_tensor.device()
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)), input_tensor.dtype, ttnn.TILE_LAYOUT, device, out_mem
    )
    page_bytes = input_tensor.buffer_aligned_page_size()
    num_pages = output_tensor.buffer_num_pages()
    assert num_pages == input_tensor.buffer_num_pages()
    cores, core_ranges, assignment = _line(device, placement, num_cores, num_pages)

    cb = ttnn.CBDescriptor(
        total_size=2 * block * page_bytes,  # double-buffered reader -> writer
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_ID, data_format=input_tensor.dtype, page_size=page_bytes)
        ],
    )

    reader_ct = [page_bytes, kernel_iters, block] + list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct = [page_bytes, kernel_iters, block, write_split] + list(
        ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    )

    in_addr, out_addr = input_tensor.buffer_address(), output_tensor.buffer_address()
    page_range = {(c.x, c.y): a for c, a in zip(cores, assignment)}

    # swap_columns: split the cores into the usual-pairing group and the swapped group (same kernels,
    # same per-core page ranges; only the NoC each kernel issues on differs between the groups)
    swapped = _swapped_columns(device, swap_columns, swap_anchor) if swap_columns else set()
    groups = [(noc, [c for c in cores if c.x not in swapped]), (_OTHER[noc], [c for c in cores if c.x in swapped])]

    kernels = []
    for read_noc, group in groups:
        if not group:
            continue
        group_ranges = _core_range_set(group)
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core in group:
            start_page, count = page_range[(core.x, core.y)]
            reader_rt[core.x][core.y] = [in_addr, start_page, count]
            writer_rt[core.x][core.y] = [out_addr, start_page, count]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "copy_reader.cpp"),
                core_ranges=group_ranges,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=_dm_config(ttnn.DataMovementProcessor.RISCV_1, read_noc, noc_mode),  # reads on `read_noc`
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "copy_writer.cpp"),
                core_ranges=group_ranges,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                # writes on the other NoC (every write_split-th page on the read NoC); both kernels on a
                # Tensix core must share one NOC_MODE, so the reader follows the writer into dynamic mode
                config=_dm_config(ttnn.DataMovementProcessor.RISCV_0, _OTHER[read_noc], noc_mode),
            )
        )
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb])
    return ttnn.generic_op([input_tensor, output_tensor], program)


def noc_placement(
    input_tensor: ttnn.Tensor,
    *,
    op: str = "copy",
    noc: str = "noc0",
    placement: str = "row",
    num_cores: int = None,
    kernel_iters: int = 1,
    block: int = 16,
    memory_config: ttnn.MemoryConfig = None,
    write_split: int = 0,
    noc_mode: str = None,
    swap_columns=0,
    swap_anchor: str = "first",
) -> ttnn.Tensor:
    """Run one cell of the placement × NoC × operation matrix.

    Args:
        op: "read" (DRAM->L1 bench), "write" (L1->DRAM bench), or "copy" (identity copy).
        noc: "noc0" or "noc1" -- the stream's NoC; for "copy" this is the read NoC and
            writes use the other NoC.
        placement: "column" (the split_work_to_cores default trap), "row", "diagonal", or
            "grid" (the first num_cores Tensix cores of the worker grid, row-major).
        num_cores: line length; defaults to min(grid.x, grid.y) so a diagonal fits (for "grid":
            every Tensix core of the worker grid).
        kernel_iters: in-kernel repeat of the range (1 = single pass; larger = steady state).
        block: pages issued per NoC barrier (outstanding transactions; larger = more NoC pressure).
        memory_config: output placement for op="copy" (default DRAM interleaved; an L1-interleaved
            config makes the writes land in Tensix-core L1 instead of DRAM).
        write_split: op="copy" only. 0 = all writes on the write NoC; k >= 1 = every k-th output
            page is written on the read NoC instead (1/k of the writes move NoC).
        noc_mode: "dedicated" (default when write_split == 0) or "dynamic" (required, and the
            default, when write_split > 0). Applies to both copy kernels.
        swap_columns: op="copy" only. 0 = the usual pairing everywhere; N = the Tensix cores in the N
            worker-grid columns nearest DRAM (queried from the device) read on the other NoC and write
            on `noc`; "all" = every column (the fully swapped pairing).
        swap_anchor: "first" (rank columns by distance from the DRAM column next to bank 0) or "every"
            (by distance from the nearest DRAM column).

    Returns the copy output for op="copy" (== input); for the benches the returned tensor's
    contents are meaningless -- the point is the isolated read/write NoC traffic.
    """
    if op not in OPS:
        raise ValueError(f"noc_placement example: op must be one of {OPS}, got {op!r}")
    if noc not in NOCS:
        raise ValueError(f"noc_placement example: noc must be one of {NOCS}, got {noc!r}")
    if placement not in ALL_PLACEMENTS:
        raise ValueError(f"noc_placement example: placement must be one of {ALL_PLACEMENTS}, got {placement!r}")
    if block < 1:
        raise ValueError(f"noc_placement example: block must be >= 1, got {block}")
    if write_split < 0:
        raise ValueError(f"noc_placement example: write_split must be >= 0, got {write_split}")
    if write_split and op != "copy":
        raise ValueError("noc_placement example: write_split applies to op='copy' only")
    if noc_mode is None:
        noc_mode = "dynamic" if write_split else "dedicated"
    if noc_mode not in NOC_MODES:
        raise ValueError(f"noc_placement example: noc_mode must be one of {NOC_MODES}, got {noc_mode!r}")
    if write_split and noc_mode != "dynamic":
        raise ValueError("noc_placement example: write_split > 0 issues on both NoCs and needs noc_mode='dynamic'")
    if noc_mode != "dedicated" and op != "copy":
        raise ValueError("noc_placement example: noc_mode applies to op='copy' only")
    if swap_columns != "all" and (not isinstance(swap_columns, int) or swap_columns < 0):
        raise ValueError(f"noc_placement example: swap_columns must be an int >= 0 or 'all', got {swap_columns!r}")
    if swap_columns and op != "copy":
        raise ValueError("noc_placement example: swap_columns applies to op='copy' only")
    if swap_anchor not in SWAP_ANCHORS:
        raise ValueError(f"noc_placement example: swap_anchor must be one of {SWAP_ANCHORS}, got {swap_anchor!r}")
    validate(input_tensor)

    if op == "copy":
        return _copy(
            input_tensor,
            noc=noc,
            placement=placement,
            num_cores=num_cores,
            kernel_iters=kernel_iters,
            block=block,
            memory_config=memory_config,
            write_split=write_split,
            noc_mode=noc_mode,
            swap_columns=swap_columns,
            swap_anchor=swap_anchor,
        )
    return _bench(
        input_tensor,
        op=op,
        noc=noc,
        placement=placement,
        num_cores=num_cores,
        kernel_iters=kernel_iters,
        block=block,
    )
