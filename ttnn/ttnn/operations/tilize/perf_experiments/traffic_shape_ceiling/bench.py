# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""`traffic_shape_ceiling` — the bare-traffic bench for the tilize focus shape.

A STANDALONE `ttnn.ProgramDescriptor` (it never imports or calls the tilize op)
that reproduces, transfer for transfer, the NoC traffic the tilize focus shape
`[1,1,32,16384]` bf16 RM DRAM -> TILE DRAM puts on a 64-core Wormhole B0 grid,
with the tilize itself and everything else deleted. Its only purpose is to
answer "what is the ceiling for THIS traffic shape?" by removing the one thing
the op cannot remove — the read -> write dependency — and, separately, the one
thing the op cannot change — the 512-B read transaction.

THE TRAFFIC BEING REPRODUCED (per core, exactly one wave, block_id = the core's
row-wise raster index 0..63, which is the assignment `split_work_to_cores(grid,
64, row_wise=True)` hands the op at R=1, C=512, block_width_tiles=8):

  reads  (NoC0): 32 x 512 B behind ONE barrier, from the 32 pages (32768 B each)
                 of a `[1,1,32,16384]` bf16 ROW_MAJOR DRAM tensor, at byte
                 offset `512 * block_id` inside every page, issued in the op's
                 core-rotated order (start at row `block_id % 32`).
  writes (NoC1): 8 x 2048 B behind ONE barrier, whole pages `8*block_id + i`
                 of a 512-page x 2048 B interleaved DRAM buffer, issued in the
                 op's column-rotated order (start at `block_id % 8`).

WHY THE DESTINATION IS ROW_MAJOR AND NOT TILE. The op's output is a
`[1,1,32,16384]` bf16 TILE interleaved DRAM tensor: 512 pages of 2048 B. A
`[1,1,512,1024]` bf16 ROW_MAJOR interleaved DRAM tensor is 512 pages of 2048 B
as well — SAME page count, SAME page size, so the same interleaved page ->
bank round-robin and byte-for-byte the same NoC traffic — but its bytes come
back through `ttnn.to_torch` in page order instead of through the tile
de-swizzle, which is what makes the transfers value-checkable (see VERIFICATION
below). The 2-KB read source is the same trick.

THE RUNGS (all seven run in ONE process, interleaved rep by rep, so process-level
drift cannot be mistaken for a variant difference):

  chained            32x512 B reads -> CB -> 8x2048 B writes. The op's dependency
                     with compute deleted. Sanity target ~11.4 us (op 12.36 us
                     minus its 951 ns compute).
  chained_2k          8x2048 B reads -> CB -> 8x2048 B writes. Same dependency,
                     4x wider read transaction, same bytes.
  independent        the SAME 32x512 B reads and the SAME 8x2048 B writes, but
                     the writes source a DIFFERENT, never-produced L1 buffer, so
                     there is no ordering relation whatsoever between the two
                     streams and they coexist from kernel start. THE CEILING for
                     this traffic shape.
  independent_2k     ditto with 8x2048 B reads. Isolates H1 (transaction shape)
                     from H2 (dependency).
  reads_only_512 /
  reads_only_2k /
  writes_only        one-sided calibration rungs (`num_writes = 0` /
                     `read_enabled = 0`). Both kernels are still instantiated on
                     every rung, so the dispatch/launch floor is a constant.

VERIFICATION OF THE TRAFFIC (values are NOT the gate on the dependency-free
rungs — the payload there is garbage by design — but the TRANSFERS are):

  (a) HOST ARG-TABLE PARTITION. `_plan()` builds the exact per-core (page,
      byte offset, length) triples and `_assert_traffic_partition` asserts they
      TILE their tensor exactly once: the 2048 read triples of the 512-B shape
      cover each of the 32 x 32768 B source pages with 64 disjoint 512-B slices,
      the 512 read triples of the 2-KB shape cover each 2048-B page once, and
      the 512 write triples hit each destination page exactly once. Total
      1,048,576 B read and 1,048,576 B written per rung, matching the op.
  (b) VALUE GATE ON THE CHAINED RUNGS. `chained` and `chained_2k` run the
      IDENTICAL reader compile-time/runtime args as their `independent` twins
      (only `do_push`/`use_cb` differ, an `if constexpr` around the CB calls),
      and their destination is asserted bit-exact against the byte permutation
      the traffic implies. So the read addresses, the read lengths, the L1
      landing offsets and the write page ids are all pinned by a torch equality.
  (c) MARKER GATE ON THE SCRATCH-SOURCED RUNGS. `independent`, `independent_2k`
      and `writes_only` stamp a (block_id, i) marker into the first and the LAST
      4 bytes of each 2048-B scratch page; the host asserts destination page
      `8*block_id + i` carries exactly that marker at both ends. That pins the
      write page id, the 2048-B length and the L1 source stride on precisely the
      rungs where no value check is possible.
  (d) CROSS-CHECK AGAINST THE OP. `reads_only_512` and `writes_only` should
      reproduce the op's own payload-ablation numbers (5420 ns / 7382 ns on this
      tree); they are the independent confirmation that the reconstruction is
      the op's traffic and not a lookalike.

Precision contract: bf16 in, bf16 out, untouched — there is no compute kernel in
this bench at all, so there is nothing to tune.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --- the focus shape's traffic constants (Wormhole B0 n150, 8x8 grid) --------
NUM_CORES = 64
GRID_X = 8
RM_SHAPE = (1, 1, 32, 16384)  # the op's input: 32 pages x 32768 B
PAGED_SHAPE = (1, 1, 512, 1024)  # 512 pages x 2048 B == the op's TILE output geometry
READ_BYTES_512 = 512  # block_width_tiles(8) * 32 * 2 B
NUM_READS_512 = 32  # tile_h
READ_BYTES_2K = 2048
NUM_READS_2K = 8
WRITE_BYTES = 2048  # out_tile_bytes
NUM_WRITES = 8  # block_width_tiles
BYTES_PER_CORE = 16384

CB_IN = 0
CB_SCRATCH = 1
CB_PAGES = 8
CB_PAGE_BYTES = 2048

# variant -> (read_mode, chained, write_enabled)
#   read_mode: "512" | "2k" | None
VARIANTS = {
    "chained": ("512", True, True),
    "chained_2k": ("2k", True, True),
    "independent": ("512", False, True),
    "independent_2k": ("2k", False, True),
    "reads_only_512": ("512", False, False),
    "reads_only_2k": ("2k", False, False),
    "writes_only": (None, False, True),
}
VARIANT_ORDER = list(VARIANTS)


def _cores():
    """block_id k -> core (k % 8, k // 8): the row-wise raster order
    `split_work_to_cores(grid, 64, row_wise=True)` gives the op at 64 blocks."""
    return [ttnn.CoreCoord(k % GRID_X, k // GRID_X) for k in range(NUM_CORES)]


def _core_range_set():
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, NUM_CORES // GRID_X - 1))}
    )


def _plan(read_mode, write_enabled):
    """Per-core traffic table: (read triples, write triples) as (page, offset, length)."""
    reads, writes = [], []
    for b in range(NUM_CORES):
        if read_mode == "512":
            reads.append([(r, READ_BYTES_512 * b, READ_BYTES_512) for r in range(NUM_READS_512)])
        elif read_mode == "2k":
            reads.append([(NUM_READS_2K * b + i, 0, READ_BYTES_2K) for i in range(NUM_READS_2K)])
        else:
            reads.append([])
        writes.append([(NUM_WRITES * b + i, 0, WRITE_BYTES) for i in range(NUM_WRITES)] if write_enabled else [])
    return reads, writes


def assert_traffic_partition(read_mode, write_enabled, reads, writes):
    """(a) The traffic table TILES its tensor exactly once — no gap, no overlap."""
    if read_mode is not None:
        flat = [t for per_core in reads for t in per_core]
        n_pages, page_bytes = (32, 32768) if read_mode == "512" else (512, 2048)
        assert len(flat) == NUM_CORES * (NUM_READS_512 if read_mode == "512" else NUM_READS_2K)
        assert len({(p, o) for p, o, _ in flat}) == len(flat), "read triples overlap"
        assert sum(n for _, _, n in flat) == n_pages * page_bytes, "reads do not cover the source exactly"
        cover = {}
        for p, o, n in flat:
            cover.setdefault(p, []).append((o, n))
        assert len(cover) == n_pages, f"reads touch {len(cover)} of {n_pages} source pages"
        for p, segs in cover.items():
            segs.sort()
            at = 0
            for o, n in segs:
                assert o == at, f"gap/overlap in source page {p} at byte {o} (expected {at})"
                at += n
            assert at == page_bytes, f"source page {p} covered {at} of {page_bytes} B"
        assert sum(n for _, _, n in flat) == NUM_CORES * BYTES_PER_CORE
    if write_enabled:
        flat = [t for per_core in writes for t in per_core]
        assert len(flat) == NUM_CORES * NUM_WRITES
        assert {p for p, _, _ in flat} == set(range(512)), "writes do not hit each destination page exactly once"
        assert all(n == WRITE_BYTES and o == 0 for _, o, n in flat)
        assert sum(n for _, _, n in flat) == NUM_CORES * BYTES_PER_CORE


def create_program_descriptor(variant, src_rm, src_paged, dst):
    read_mode, chained, write_enabled = VARIANTS[variant]
    reads, writes = _plan(read_mode, write_enabled)
    assert_traffic_partition(read_mode, write_enabled, reads, writes)

    src = src_rm if read_mode == "512" else src_paged
    num_reads = 0 if read_mode is None else (NUM_READS_512 if read_mode == "512" else NUM_READS_2K)
    read_bytes = READ_BYTES_512 if read_mode == "512" else READ_BYTES_2K
    num_writes = NUM_WRITES if write_enabled else 0

    cores = _cores()
    crs = _core_range_set()

    def cb(index):
        return ttnn.CBDescriptor(
            total_size=CB_PAGES * CB_PAGE_BYTES,
            core_ranges=crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=CB_PAGE_BYTES)
            ],
        )

    reader_ct = [CB_IN, CB_PAGES, 1 if read_mode is not None else 0, 1 if chained else 0, max(num_reads, 1), read_bytes]
    reader_ct.extend(ttnn.TensorAccessorArgs(src).get_compile_time_args())
    writer_ct = [
        CB_IN,
        CB_SCRATCH,
        1 if chained else 0,
        CB_PAGES,
        num_writes,
        WRITE_BYTES,
        0 if chained else 1,  # verify_fill: only where the payload is garbage
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(dst).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    src_addr = src.buffer_address()
    dst_addr = dst.buffer_address()
    for b, core in enumerate(cores):
        if read_mode == "512":
            page_base, byte_off, rot = 0, READ_BYTES_512 * b, b % NUM_READS_512
        elif read_mode == "2k":
            page_base, byte_off, rot = NUM_READS_2K * b, 0, b % NUM_READS_2K
        else:
            page_base, byte_off, rot = 0, 0, 0
        # The kernel's own (page_base + idx, byte_off, read_bytes) must be the
        # traffic table this file just partition-checked.
        if read_mode is not None:
            assert sorted(reads[b]) == sorted((page_base + i, byte_off, read_bytes) for i in range(num_reads))
        reader_rt[core.x][core.y] = [src_addr, page_base, byte_off, rot]
        writer_rt[core.x][core.y] = [dst_addr, NUM_WRITES * b, b % NUM_WRITES, b]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tsc_reader.cpp"),
        core_ranges=crs,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),  # NCRISC / NoC0, as the op's reader
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tsc_writer.cpp"),
        core_ranges=crs,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),  # BRISC / NoC1, as the op's writer
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=[cb(CB_IN), cb(CB_SCRATCH)])


def run(variant, src_rm, src_paged, dst):
    pd = create_program_descriptor(variant, src_rm, src_paged, dst)
    return ttnn.generic_op([src_rm, src_paged, dst], pd)
