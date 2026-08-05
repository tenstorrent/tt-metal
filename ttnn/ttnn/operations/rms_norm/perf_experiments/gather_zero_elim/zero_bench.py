# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED PRICE of rms_norm's gather boot-zeroing (`writer_gather_zero`), and of
every alternative to DELETING it.

THE STAGE, in isolation
-----------------------
Once, before the combine loop, the group ROOT zeroes exactly the bytes of
`cb_partials_gathered` the gather never writes:
  * at GATHER_FACES == 2, faces 1 and 3 of every page (1024 B each of a 4096 B fp32
    tile page);
  * at GATHER_FACES == 3, face 3 only;
  * at odd GROUP_SIZE, the whole pad page of every tile-row.
Zeroing the WHOLE CB instead is already a known-wrong answer for the op (it wipes
members that have already landed -- measured pcc 0.87-0.99), so `whole_cb` below is
priced only as the cost half of a hypothetical "zero everything up front behind a
readiness handshake", never as a drop-in.

WHY IT COSTS WHAT IT COSTS.  `Noc::async_write_zeros` (local-L1 overload) is a CHUNKED
NoC READ LOOPBACK from MEM_ZEROS_BASE, and MEM_ZEROS_SIZE is 512 B on both Wormhole and
Blackhole (tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl + .../blackhole/dev_mem_map.h).
So one 1024 B face zero is TWO NoC transactions, and the focus shape's 64-page CB at
GATHER_FACES == 2 is 64 x 2 x 2 = 256 transactions.  The stage is transaction-ISSUE
bound, not byte bound -- which is what makes `scratch` (below) a real lever.

THE MENU
--------
  none        DELETION.  Nothing is zeroed.  Also the ABLATION FLOOR the other
              variants are subtracted against (launch + CB scaffolding only).
  faces       THE OP TODAY (rms_norm_writer.cpp `writer_gather_zero`), verbatim: one
              `noc.async_write_zeros` per unshipped face, plus one per pad page.
              Runs on the WRITER == BRISC == NoC1, as the op does.
  pad_only    The CARVE-OUT: faces 1/3 left alone, only the odd-GROUP_SIZE pad pages
              zeroed.  BYTE-IDENTICAL TO `none` at every even GROUP_SIZE (nothing to
              zero), which is exactly why it is on the menu -- if the poison bench says
              faces 1/3 are safe but the pad is not, this is the shape of the fix.
  scratch     Same bytes as `faces`, HALF the transactions: zero a 4096 B scratch page
              ONCE, then one `noc_async_write` per unshipped face out of it.  1 txn per
              face instead of 2 (512 B chunking), and 8 txns per pad page instead of 8
              -- i.e. the pad case is unchanged and the face case halves.
  whole_cb    One `async_write_zeros` over the entire CB.  Fewest API calls, MOST bytes
              (PAGES * 4096), and UNSAFE in the op without an extra readiness handshake.
  faces_r     `faces`, but issued from the READER (NCRISC / NoC0) instead of the writer.
              On the focus shape the reader is essentially idle (`reader_read_x` = 56 ns
              total -- the input is a native zero-copy shard CB), so this prices the
              boot on the idle NoC half.  Its WALL value in the op depends on overlap
              and is NOT claimed from this bench.
  scratch_r   `scratch` on the reader.

CORRECTNESS IS THE ONLY PASS/FAIL, and it is BYTE-EXACT here.  The gather CB is backed
by a resident fp32 shard that the host PRE-FILLS with a distinctive non-zero pattern.
After the run the host checks, byte for byte, that (a) exactly the bytes the variant
promises are zero and (b) every other byte is UNCHANGED.  That is what disqualifies a
scheme that races or overshoots (the op's own `writer_gather_zero` comment records a
0.87-0.99 pcc from exactly such an overshoot).

Perf is MEASURED, never asserted.  `stage_ns` = this launch's DEVICE KERNEL DURATION
minus the `none` launch's for the same geometry.

------------------------------------------------------------------------------
MEASURED  (blackhole p150b, 1350 MHz, one core, ONE fresh-cache profiled run per launch;
every variant BYTE-EXACT).  Full tables: measurements.txt sections B and C.
------------------------------------------------------------------------------
FOCUS  GROUP_SIZE 8, BLOCK_ROWS 8, GATHER_FACES 2 -> 64 pages, 128 face zeros

    variant       stage_ns   txns  ns/txn      x      note
    none                 0      0       -       -     DELETION (and the floor)
    pad_only            45      0       -       -     == none at EVEN GROUP_SIZE
    faces             9900    256    38.7   1.00x     <- THE OP TODAY
    scratch           2470    136    18.2   4.01x     same bytes, byte-exact
    whole_cb          7326    512    14.3   1.35x     UNSAFE without a handshake
    faces_r           9723    256    38.0   1.02x     NoC0 instead of NoC1 -> NULL
    scratch_r         2470    136    18.1   4.01x

  The stage is TRANSACTION-ISSUE bound, not byte bound: `whole_cb` moves 2x the bytes of
  `faces` in 25% less time because it sets the read state ONCE for 512 chunks, while each
  1024 B `async_write_zeros` pays its own setup for 2 chunks.  Reproducibility across four
  independent launches of this same geometry: 9813 / 9896 / 9900 / 9900 ns (<= 1%).

GATHER_FACES  (same geometry, `faces` stage_ns):  2 -> 9901,  3 -> 2997,  4 -> 64
  At GATHER_FACES 4 nothing is unshipped, so at an EVEN GROUP_SIZE the whole stage is
  INERT BY CONSTRUCTION (64 ns is the empty loop).  It does NOT remove the odd-GROUP_SIZE
  pad need: ZERO_PAD is `GATHER_SLOTS != GROUP_SIZE` and is independent of GATHER_FACES.

DOMAIN, `faces` stage_ns (== what deletion saves off this stage), GROUP_SIZE x rows:
    rows\G      4        8        9*      28       32
    1         695     1341     1056     4417     5068
    8        5028     9896     8195    34496    39341
    32      19734    39362    32433       --       --
  `scratch` at the same points: 421 / 558 / 685 / 1250 / 1400 (rows 1), 1380 / 2468 /
  3558 / 7958 / 9048 (rows 8), 4652 / 9002 / 13361 (rows 32).  `pad_only` is 42-79 ns
  (i.e. `none`) at every EVEN GROUP_SIZE.   (*) 9 is the only ODD one.

ODD GROUP_SIZE 9 -- the pad, three ways (stage_ns):
    rows   faces   pad_only   pad_faces02
    1        962        143           159
    8       8192       1781          1959
    32     32433       6783          7518
  `pad_faces02` -- zeroing only the pad's SHIPPED faces (2048 B instead of 4096), which
  the poison bench licenses -- is a MEASURED REGRESSION of 10-11%: it halves the bytes but
  DOUBLES the API calls, and this stage pays per call (one 4096 B call = 1 setup + 8
  chunks; two 1024 B calls = 2 setups + 4 chunks).  Keep the whole-page pad zero.
"""

import ttnn

TILE = 32

CB_PART = 0  # fp32 gather CB, resident shard == cb_partials_gathered
CB_SCRATCH = 1  # one fp32 page of zeros, for the `scratch` variants

VARIANTS = ("none", "faces", "pad_only", "pad_faces02", "scratch", "whole_cb", "faces_r", "scratch_r")
_METHOD = {
    "none": 0,
    "faces": 1,
    "pad_only": 2,
    "scratch": 3,
    "whole_cb": 4,
    "pad_faces02": 5,
    "faces_r": 1,
    "scratch_r": 3,
}
# Which RISC issues the boot: the writer (BRISC / NoC1, as the op does) or the reader
# (NCRISC / NoC0, the idle half on the focus shape).
_ON_READER = ("faces_r", "scratch_r")
FLOOR = "none"

# The pattern the host pre-fills the gather CB with, so "unchanged" is checkable.
FILL_MAGIC = -7.5e-3


_KERNEL = r"""
// =============================================================================
// rms_norm perf experiment: gather_zero_elim  (ISOLATED BOOT-ZERO BENCH KERNEL)
// =============================================================================
// ONE source, compiled into BOTH the reader (NCRISC/NoC0) and the writer
// (BRISC/NoC1) instance; `DO_BOOT` selects which of the two actually issues the
// zeroing, so the SAME payload can be priced on either NoC half with no other
// difference in the program.
//
// METHOD 1 (`faces`) is rms_norm_writer.cpp's `writer_gather_zero` block VERBATIM --
// the honest baseline.  It is copied rather than re-derived so the transaction count,
// the API and the barrier are bit-identical to the op's.
// =============================================================================
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t DO_BOOT = get_compile_time_arg_val(0);
    constexpr uint32_t METHOD = get_compile_time_arg_val(1);
    constexpr uint32_t PAGES = get_compile_time_arg_val(2);         // GATHER_SLOTS * BLOCK_ROWS
    constexpr uint32_t GATHER_SLOTS = get_compile_time_arg_val(3);  // GROUP_SIZE rounded up to even
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(4);
    constexpr uint32_t GATHER_FACES = get_compile_time_arg_val(5);

    constexpr uint32_t cb_part = 0, cb_scratch = 1;
    constexpr bool ZERO_PAD = (GATHER_SLOTS != GROUP_SIZE);
    static_assert(GATHER_FACES >= 2 && GATHER_FACES <= 4, "GATHER_FACES must be 2, 3 or 4");

    if constexpr (DO_BOOT == 0 || METHOD == 0) {
        return;  // `none`: the deletion, and the ablation floor
    }

    const uint32_t stat_bytes = get_tile_size(cb_part);
    const uint32_t face_bytes = stat_bytes / 4;
    Noc noc;

    MaybeDeviceZoneScope("bench_gather_zero");
    if constexpr (METHOD == 1 || METHOD == 2) {
        // ---- the op today (METHOD 1), and the pad-only carve-out (METHOD 2) -----
        // Zeroing exactly the UNSHIPPED faces (and nothing else) is what makes this
        // race-free: a member's partial can land at any time and only ever touches
        // faces the root leaves alone.
        constexpr bool DO_FACES = (METHOD == 1);
        DataflowBuffer gather_dfb(cb_part);
        for (uint32_t p = 0; p < PAGES; ++p) {
            const uint32_t base = p * stat_bytes;
            if constexpr (ZERO_PAD) {
                if (p % GATHER_SLOTS >= GROUP_SIZE) {  // a pad slot: zero it whole
                    noc.async_write_zeros(gather_dfb, stat_bytes, {.offset_bytes = base});
                    continue;
                }
            }
            if constexpr (DO_FACES && GATHER_FACES == 2) {  // faces 1 and 3 unshipped
                noc.async_write_zeros(gather_dfb, face_bytes, {.offset_bytes = base + face_bytes});
            }
            if constexpr (DO_FACES && GATHER_FACES < 4) {
                noc.async_write_zeros(gather_dfb, face_bytes, {.offset_bytes = base + 3 * face_bytes});
            }
        }
        noc.write_zeros_l1_barrier();
    } else if constexpr (METHOD == 3) {
        // ---- `scratch`: the same bytes, half the transactions -------------------
        // `async_write_zeros` chunks at MEM_ZEROS_SIZE (512 B), so a 1024 B face costs
        // TWO NoC transactions.  Zeroing one 4096 B scratch page up front and writing
        // 1024 B out of it costs ONE transaction per face.  The barrier between the two
        // is MANDATORY: the zeros overload owns the write command buffer until
        // write_zeros_l1_barrier() returns it (noc.h's zero-mode note).
        DataflowBuffer scratch_dfb(cb_scratch);
        noc.async_write_zeros(scratch_dfb, stat_bytes);
        noc.write_zeros_l1_barrier();
        const uint32_t src = get_write_ptr(cb_scratch);
        const uint32_t dst_base = get_write_ptr(cb_part);
        for (uint32_t p = 0; p < PAGES; ++p) {
            const uint32_t base = dst_base + p * stat_bytes;
            if constexpr (ZERO_PAD) {
                if (p % GATHER_SLOTS >= GROUP_SIZE) {
                    noc_async_write(src, get_noc_addr(base), stat_bytes);
                    continue;
                }
            }
            if constexpr (GATHER_FACES == 2) {
                noc_async_write(src, get_noc_addr(base + face_bytes), face_bytes);
            }
            if constexpr (GATHER_FACES < 4) {
                noc_async_write(src, get_noc_addr(base + 3 * face_bytes), face_bytes);
            }
        }
        noc_async_write_barrier();
    } else if constexpr (METHOD == 5) {
        // ---- `pad_faces02`: pad_only, MINIMISED by measurement -------------------
        // The poison bench measured that a pad page whose faces 0/2 are exactly zero is
        // BIT-IDENTICAL to a fully zeroed pad page even with faces 1/3 poisoned
        // (`pad_seed=faces13`).  So only the pad's SHIPPED faces have to be defined:
        // 2048 B per pad page instead of 4096, 4 zero-chunks instead of 8.
        if constexpr (ZERO_PAD) {
            DataflowBuffer gather_dfb(cb_part);
            for (uint32_t p = 0; p < PAGES; ++p) {
                if (p % GATHER_SLOTS >= GROUP_SIZE) {
                    const uint32_t base = p * stat_bytes;
                    noc.async_write_zeros(gather_dfb, face_bytes, {.offset_bytes = base});
                    noc.async_write_zeros(gather_dfb, face_bytes, {.offset_bytes = base + 2 * face_bytes});
                }
            }
            noc.write_zeros_l1_barrier();
        }
    } else {
        // ---- `whole_cb`: one call, PAGES * 4096 bytes ---------------------------
        // NOT a drop-in for the op: it wipes members that already landed (the op's
        // recorded 0.87-0.99 pcc).  Priced as the cost half of a hypothetical
        // zero-everything-behind-a-readiness-handshake scheme.
        DataflowBuffer gather_dfb(cb_part);
        noc.async_write_zeros(gather_dfb, PAGES * stat_bytes);
        noc.write_zeros_l1_barrier();
    }
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _sharded(h_tiles):
    return ttnn.create_sharded_memory_config(
        shape=(h_tiles * TILE, TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def gather_slots(group_size):
    return group_size + group_size % 2


def _cb(index, page_size, num_pages, data_format):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=_single_core(),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def expected_zeros(variant, group_size, rows, gather_faces):
    """The regions this variant PROMISES to zero, as (page, row_lo, row_hi, col_lo, col_hi).

    A tile's 2x2 face grid in the LOGICAL view: face f covers rows 16*(f//2)..+16 and
    columns 16*(f%2)..+16.  So faces 1 and 3 are columns 16..31 (all rows) and face 3
    alone is rows 16..31 of columns 16..31 -- which is what makes the byte-exact host
    check expressible on the torch view of the tensor.
    """
    gp = gather_slots(group_size)
    regions = []
    if variant == "none":
        return regions
    if variant == "whole_cb":
        return [(p, 0, TILE, 0, TILE) for p in range(gp * rows)]
    do_faces = variant in ("faces", "faces_r", "scratch", "scratch_r")
    for p in range(gp * rows):
        if (p % gp) >= group_size:  # a pad page is folded WHOLE
            if variant == "pad_faces02":  # ... but only its SHIPPED faces are read
                regions.append((p, 0, TILE, 0, 16))
            else:
                regions.append((p, 0, TILE, 0, TILE))
        elif do_faces:
            if gather_faces == 2:  # faces 1 and 3
                regions.append((p, 0, TILE, 16, TILE))
            elif gather_faces == 3:  # face 3 only
                regions.append((p, 16, TILE, 16, TILE))
    return regions


def run_variant(device, variant, group_size, rows, *, gather_faces=2):
    """Run one variant once on device; returns (metrics, byte-exact correctness verdict)."""
    import torch

    if variant not in VARIANTS:
        raise ValueError(f"gather_zero_elim: variant must be one of {VARIANTS}, got {variant!r}")

    gp = gather_slots(group_size)
    pages = gp * rows
    fill = torch.full((pages * TILE, TILE), FILL_MAGIC, dtype=torch.float32)
    part_dev = ttnn.from_torch(
        fill, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_sharded(pages)
    )
    ft = ttnn.tile_size(ttnn.float32)

    on_reader = variant in _ON_READER
    method = _METHOD[variant]
    ct = [0, method, pages, gp, group_size, gather_faces]

    def kdesc(do_boot, config):
        return ttnn.KernelDescriptor(
            kernel_source=_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_single_core(),
            compile_time_args=[1 if do_boot else 0] + ct[1:],
            config=config,
        )

    reader = kdesc(on_reader, ttnn.ReaderConfigDescriptor())
    writer = kdesc(not on_reader, ttnn.WriterConfigDescriptor())
    descriptor = ttnn.ProgramDescriptor(
        kernels=[reader, writer],
        semaphores=[],
        cbs=[
            ttnn.cb_descriptor_from_sharded_tensor(CB_PART, part_dev),
            _cb(CB_SCRATCH, ft, 1, ttnn.float32),
        ],
    )
    # `generic_op` requires >= 2 io_tensors (one input, one output).  This stage has
    # exactly one tensor -- the gather CB it zeroes in place -- so a one-tile scratch
    # tensor rides along as the nominal output.  No kernel and no CB touches it, so it
    # cannot affect the measurement.
    sink = ttnn.allocate_tensor_on_device(ttnn.Shape([TILE, TILE]), ttnn.float32, ttnn.TILE_LAYOUT, device, _sharded(1))
    ttnn.generic_op([part_dev, sink], descriptor)
    got = ttnn.to_torch(part_dev).to(torch.float32).reshape(pages, TILE, TILE)

    # ---- byte-exact gate: exactly the promised bytes zero, everything else intact ----
    regions = expected_zeros(variant, group_size, rows, gather_faces)
    want = torch.full((pages, TILE, TILE), FILL_MAGIC, dtype=torch.float32)
    for p, r_lo, r_hi, c_lo, c_hi in regions:
        want[p, r_lo:r_hi, c_lo:c_hi] = 0.0
    exact = bool(torch.equal(got, want))
    mismatch = int((got != want).sum())

    return {
        "variant": variant,
        "group_size": group_size,
        "gather_slots": gp,
        "rows": rows,
        "gather_faces": gather_faces,
        "pages": pages,
        "on_reader": on_reader,
        "bytes_zeroed": sum((r_hi - r_lo) * (c_hi - c_lo) * 4 for _, r_lo, r_hi, c_lo, c_hi in regions),
        "byte_exact": exact,
        "mismatched_elems": mismatch,
        "ablation": variant == FLOOR,
    }
