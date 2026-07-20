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
OPS = ("read", "write", "copy")
NOCS = ("noc0", "noc1")

_NOC = {"noc0": ttnn.NOC.NOC_0, "noc1": ttnn.NOC.NOC_1}
_OTHER = {"noc0": "noc1", "noc1": "noc0"}


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


def _ordered_cores(placement, n):
    """The N cores of a placement, in a fixed order so page assignment is identical
    across placements (core index k always gets the same page range; only its
    physical (x, y) differs)."""
    if placement == "row":
        return [ttnn.CoreCoord(i, 0) for i in range(n)]
    if placement == "column":
        return [ttnn.CoreCoord(0, i) for i in range(n)]
    if placement == "diagonal":
        return [ttnn.CoreCoord(i, i) for i in range(n)]
    raise ValueError(f"noc_placement example: placement must be one of {PLACEMENTS}, got {placement!r}")


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


def _resolve_num_cores(device, num_cores):
    """Cap at min(grid.x, grid.y) so a diagonal of this length fits on the grid."""
    grid = device.compute_with_storage_grid_size()
    max_line = min(grid.x, grid.y)
    if num_cores is None:
        return max_line
    if num_cores < 1 or num_cores > max_line:
        raise ValueError(
            f"noc_placement example: num_cores must be in [1, {max_line}] "
            f"(a diagonal must fit on the {grid.x}x{grid.y} grid), got {num_cores}"
        )
    return num_cores


def _dm_config(processor, noc):
    return ttnn.DataMovementConfigDescriptor(processor=processor, noc=_NOC[noc])


def _line(device, placement, num_cores, min_pages):
    """Resolve the placement into (cores, core_ranges, assignment, page_bytes)."""
    num_cores = _resolve_num_cores(device, num_cores)
    if min_pages < num_cores:
        raise ValueError(f"noc_placement example: need >= {num_cores} pages, got {min_pages}")
    cores = _ordered_cores(placement, num_cores)
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


def _copy(input_tensor, *, noc, placement, num_cores, kernel_iters, block, memory_config):
    """Identity DRAM->DRAM copy: reader on `noc`, writer on the other NoC (so the read
    and write streams never share links). Output equals input for every placement/NoC."""
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
    writer_ct = [page_bytes, kernel_iters, block] + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    in_addr, out_addr = input_tensor.buffer_address(), output_tensor.buffer_address()
    for core, (start_page, count) in zip(cores, assignment):
        reader_rt[core.x][core.y] = [in_addr, start_page, count]
        writer_rt[core.x][core.y] = [out_addr, start_page, count]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "copy_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=_dm_config(ttnn.DataMovementProcessor.RISCV_1, noc),  # reads on `noc`
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "copy_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=_dm_config(ttnn.DataMovementProcessor.RISCV_0, _OTHER[noc]),  # writes on the other NoC
    )
    program = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=[cb])
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
) -> ttnn.Tensor:
    """Run one cell of the placement × NoC × operation matrix.

    Args:
        op: "read" (DRAM->L1 bench), "write" (L1->DRAM bench), or "copy" (identity copy).
        noc: "noc0" or "noc1" -- the stream's NoC; for "copy" this is the read NoC and
            writes use the other NoC.
        placement: "column" (the split_work_to_cores default trap), "row", or "diagonal".
        num_cores: line length; defaults to min(grid.x, grid.y) so a diagonal fits.
        kernel_iters: in-kernel repeat of the range (1 = single pass; larger = steady state).
        block: pages issued per NoC barrier (outstanding transactions; larger = more NoC pressure).

    Returns the copy output for op="copy" (== input); for the benches the returned tensor's
    contents are meaningless -- the point is the isolated read/write NoC traffic.
    """
    if op not in OPS:
        raise ValueError(f"noc_placement example: op must be one of {OPS}, got {op!r}")
    if noc not in NOCS:
        raise ValueError(f"noc_placement example: noc must be one of {NOCS}, got {noc!r}")
    if placement not in PLACEMENTS:
        raise ValueError(f"noc_placement example: placement must be one of {PLACEMENTS}, got {placement!r}")
    if block < 1:
        raise ValueError(f"noc_placement example: block must be >= 1, got {block}")
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
