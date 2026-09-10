# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Barrier-granularity example: keep DRAM reads in flight ACROSS the wait.

`trid_double_issue` is an identity copy of an interleaved DRAM tensor, run as a
two-kernel pipeline: a reader (NCRISC, NoC0) pulls pages into a circular buffer
in blocks, and a writer (BRISC, NoC1) drains that CB straight back to DRAM.
There is no compute kernel, so the math engines are out of the measurement.

The single idea: WHICH BARRIER RETIRES A BLOCK.

  variant="full_barrier" (baseline)
      Issue a block of async reads, then `noc_async_read_barrier()` — which waits
      for EVERY outstanding read on the NoC. When it returns, nothing is in
      flight. The next block is not issued until after that wait, so the reader
      pays the full DRAM round trip once per block with an empty NoC underneath
      it. This is the ordinary, correct way to write the loop.

  variant="trid_double_issue"
      Tag each block with a transaction id (`noc_async_read_set_trid`) and retire
      it with `noc_async_read_barrier_with_trid(previous_id)`, which waits only
      for that id. Blocks are issued `num_trids` deep before the first is awaited,
      so while the reader blocks on block k, blocks k+1 .. k+num_trids-1 are
      already on the wire. At least one request is in flight at all times.

The read ISSUE call is `noc_async_read` in BOTH variants — the transaction id
lives in the read command buffer's NOC_PACKET_TAG register, which an ordinary
`noc_async_read` never writes, so a tag set once rides along on every subsequent
read. Nothing about the addressing changes, which is why this stays on plain
INTERLEAVED DRAM (consecutive pages in different banks) rather than needing the
single-bank `..._with_state_with_trid` form.

Everything else is held constant between the variants: same block size, same
`cb_blocks` CB depth, same page order, same cores, and the same byte-identical
writer. The baseline has its own knob, `ahead`: how many blocks it issues before its one
barrier (pushing them individually afterwards). `ahead=1` is the idiomatic loop;
`ahead>1` spends the spare CB to keep more reads in flight and is the STRONGEST a
non-trid reader can be. It cannot go further: the only non-trid completion signal
is a COUNT of finished reads (NIU_MST_RD_RESP_RECEIVED vs noc_reads_num_issued),
and read responses take dynamically-assigned VCs so they can land out of order —
a count never proves a SPECIFIC earlier block arrived, so nothing may be pushed
until the barrier has drained everything. Per-id counters are the only per-group
completion signal, which is exactly the gap trids fill.

See README.md for the mechanism, the measured ns, and the CLI.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
CB_IN = 0  # reader -> writer

# Baseline first: the full barrier is what you write by default; trids are the fix.
VARIANTS = ("full_barrier", "trid_double_issue")

# Transaction ids are a 4-bit field (0x0-0xF) and id 0 means "untagged", so the
# kernel uses ids 1..num_trids. Depths beyond a handful stop helping long before
# the hardware limit -- the point is only to never let the NoC go empty.
MAX_TRIDS = 15

# Tile formats = bytes per NoC transaction (bfloat8_b ~1088 B, bfloat16 2048 B,
# float32 4096 B). The kernels are dtype-agnostic; the CB page size is queried
# from the tensor and never hard-coded.
SUPPORTED_DTYPES = (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32)


def validate(input_tensor):
    """Keep the example dead simple: 2D, TILE, tile-aligned, interleaved DRAM."""
    shape = list(input_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"trid_double_issue example: rank must be 2, got {len(shape)}")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("trid_double_issue example: input must be TILE_LAYOUT")
    if input_tensor.dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"trid_double_issue example: dtype must be one of {SUPPORTED_DTYPES}, got {input_tensor.dtype}"
        )
    h, w = shape
    if h % TILE or w % TILE:
        raise ValueError(f"trid_double_issue example: H and W must be multiples of {TILE}, got {shape}")


def _resolve_num_cores(device, num_cores):
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    if num_cores is None:
        return 1  # one core is the cleanest picture: one read stream, no cross-core NoC contention
    if num_cores < 1 or num_cores > max_cores:
        raise ValueError(f"trid_double_issue example: num_cores must be in [1, {max_cores}], got {num_cores}")
    return num_cores


def _ordered_cores(device, n):
    """`n` cores filled row-major. Identical for both variants, so placement is held constant."""
    grid = device.compute_with_storage_grid_size()
    return [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(n)]


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _assign_pages(num_pages, n, block):
    """Contiguous page ranges by core index, handed out in WHOLE blocks.

    The trid pipeline addresses its landing slots as whole blocks offset from the
    CB write pointer, so a core's page count must be a multiple of `block` for the
    slot arithmetic to stay aligned across iterations. Any leftover (< block) goes
    to the last core, which reads it through the plain full-barrier tail path.
    Identical for both variants -> identical per-core work.
    """
    nblocks, rem = divmod(num_pages, block)
    if nblocks < n:
        raise ValueError(
            f"trid_double_issue example: need at least {n} whole blocks of {block} pages "
            f"for {n} cores, got {nblocks} ({num_pages} pages)"
        )
    base, extra = divmod(nblocks, n)
    ranges, start = [], 0
    for k in range(n):
        count = (base + (1 if k < extra else 0)) * block
        if k == n - 1:
            count += rem  # the sub-block remainder rides on the last core
        ranges.append((start, count))
        start += count
    return ranges


def create_program_descriptor(
    input_tensor, output_tensor, *, variant, num_cores, block, num_trids, cb_blocks, kernel_iters, ahead=1
):
    if variant not in VARIANTS:
        raise ValueError(f"trid_double_issue example: variant must be one of {VARIANTS}, got {variant!r}")
    device = input_tensor.device()
    num_cores = _resolve_num_cores(device, num_cores)

    # num_trids == 0 selects the baseline path in the kernel; the flag is what the
    # variant name means, so the two are bound here and nowhere else.
    trids = 0 if variant == "full_barrier" else num_trids

    page_bytes = input_tensor.buffer_aligned_page_size()  # per-tile NoC transaction size for this dtype
    num_pages = output_tensor.buffer_num_pages()
    assert num_pages == input_tensor.buffer_num_pages()

    cores = _ordered_cores(device, num_cores)
    core_ranges = _core_range_set(cores)
    assignment = _assign_pages(num_pages, num_cores, block)

    # cb_blocks is passed in and is the SAME for every variant and every num_trids
    # in a sweep, so CB depth is never a confound: only the barrier changes.
    cb_tiles = cb_blocks * block
    cb_in = ttnn.CBDescriptor(
        total_size=cb_tiles * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_IN, data_format=input_tensor.dtype, page_size=page_bytes)
        ],
    )

    reader_ct_args = [page_bytes, kernel_iters, block, trids, cb_blocks, ahead]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = [page_bytes, kernel_iters, block]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for core, (start_page, count) in zip(cores, assignment):
        reader_rt_args[core.x][core.y] = [in_addr, start_page, count]
        writer_rt_args[core.x][core.y] = [out_addr, start_page, count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "trid_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),  # reader -> NCRISC / NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "trid_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),  # writer -> BRISC / NoC1
    )

    return ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel], semaphores=[], cbs=[cb_in])


def trid_double_issue(
    input_tensor: ttnn.Tensor,
    *,
    variant: str = "trid_double_issue",
    block: int = 4,
    num_trids: int = 2,
    cb_blocks: int = 6,
    num_cores: int = None,
    kernel_iters: int = 1,
    ahead: int = 1,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """Identity copy of an interleaved DRAM tensor as a reader(NoC0) -> writer(NoC1) pipeline.

    Args:
        variant: "full_barrier" (baseline: `noc_async_read_barrier()` after every
            block, so the NoC drains once per block) or "trid_double_issue" (tag
            each block with a transaction id and wait only on the previous one, so
            reads stay in flight across the wait). Same reader source file, same
            `noc_async_read` issue call; only the barrier differs.
        block: pages read per block, i.e. how many async reads share one barrier.
            The smaller it is, the more often the baseline drains the NoC, and the
            more there is for trids to recover.
        num_trids: how many blocks may be in flight at once (>= 2). Ids run
            1..num_trids and recycle. Ignored by the "full_barrier" variant.
        cb_blocks: CB depth in blocks. Must be >= num_trids (the reader needs a
            free landing slot per in-flight block plus room for the writer to lag).
            Hold it FIXED across a comparison so CB depth is not a confound.
        num_cores: cores running the (independent) copy. Default 1 — the cleanest
            reading, with no cross-core NoC contention. A full grid is DRAM-
            bandwidth-bound, where latency hiding has nothing left to recover.
        kernel_iters: in-kernel repeat of the whole page range. 1 = per-launch
            latency, large = steady-state throughput.
        ahead: baseline only — how many blocks the "full_barrier" reader issues
            before its single barrier, then pushes individually. 1 is the
            idiomatic read-block/barrier/push loop; larger spends the spare CB to
            keep more reads in flight, and is the STRONGEST a non-trid reader can
            be. Must be <= cb_blocks. Ignored by the trid variant.

    Output is bitwise equal to the input for every setting.
    """
    if block < 1:
        raise ValueError(f"trid_double_issue example: block must be >= 1, got {block}")
    if kernel_iters < 1:
        raise ValueError(f"trid_double_issue example: kernel_iters must be >= 1, got {kernel_iters}")
    if not (1 <= ahead <= cb_blocks):
        raise ValueError(f"trid_double_issue example: ahead must be in [1, cb_blocks={cb_blocks}], got {ahead}")
    if variant == "trid_double_issue":
        if not (2 <= num_trids <= MAX_TRIDS):
            raise ValueError(f"trid_double_issue example: num_trids must be in [2, {MAX_TRIDS}], got {num_trids}")
        if cb_blocks < num_trids:
            raise ValueError(
                f"trid_double_issue example: cb_blocks ({cb_blocks}) must be >= num_trids ({num_trids}) "
                "so every in-flight block has its own landing slot"
            )
    validate(input_tensor)
    device = input_tensor.device()

    # The trid tail path leaves the CB write pointer off a slot boundary, which the
    # next iteration's slot arithmetic assumes away. One pass is always fine.
    if kernel_iters > 1 and input_tensor.buffer_num_pages() % block:
        raise ValueError(
            f"trid_double_issue example: kernel_iters > 1 needs the page count "
            f"({input_tensor.buffer_num_pages()}) to be a multiple of block ({block})"
        )

    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)), input_tensor.dtype, ttnn.TILE_LAYOUT, device, out_mem
    )
    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        variant=variant,
        num_cores=num_cores,
        block=block,
        num_trids=num_trids,
        cb_blocks=cb_blocks,
        kernel_iters=kernel_iters,
        ahead=ahead,
    )
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
