// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer for rms_norm (BRISC, NoC1).
//
// Mirror image of the reader — same (row-block, width-chunk) loop nest, same
// WT_CHUNK transaction granularity, so both NoC halves are batched the same
// way (a reader-only batching lever just moves the bottleneck across the CB):
//   * TILE build      : cb_output_tiles  -> whole output tiles
//   * ROW_MAJOR build  : cb_output_sticks -> output sticks, W*elem bytes each
//   * NATIVE_OUT      : nothing to move at all — compute packed straight into
//                       the output shard's own L1 through a zero-copy CB; the
//                       writer only takes the completion barrier.
//   * BAND (Ref. 2b)  : cb_output_sticks -> this core's own resident ROW_MAJOR
//                       shard, one band (w_real_elems elements) per stick, at the
//                       shard's own L1 stride.  Local L1, no DRAM traffic.
//
// The ROW_MAJOR path uses dataflow_kernel_lib::write_sticks_after_untilize,
// which is exactly the consumer contract of
// compute_kernel_lib::untilize<WT_CHUNK>(rows): it waits WT_CHUNK tile-sized
// pages per tile-row, writes only the VALID sticks (so trailing rows of a short
// final tile-row are never written) and only `row_bytes` of each (so the W tile
// padding is never written).
//
// Pass B runs once per row-block regardless of regime, so unlike the reader
// there is no pass loop here.
//
// ---------------------------------------------------------------------------
// COMBINE — the cross-core width combine (op_design.md section 3.4, Lamp L1/L4)
// ---------------------------------------------------------------------------
// When the cores of a group each own a width SLICE of the same rows, each core's
// sum(x^2) is a PARTIAL and must be combined.  This kernel owns the whole
// topology, per row-block:
//
//   1  every core   write its ONE COMPACT partial tile (cb_compact_handoff) into
//                   its own slot of the ROOT's cb_partials_gathered, then
//                   remote-inc the root's arrival semaphore.  Perf 3 / D27: the
//                   whole row-block travels as a single tile whose columns
//                   0..BLOCK_ROWS-1 are its per-tile-row partial sums, so this is
//                   ONE transaction per round instead of BLOCK_ROWS face-runs.
//   2  root         once arrivals reach (blk+1) * (GROUP_SIZE - 1), publish the
//                   gathered round so compute can fold + finalize it in ONE DEST
//                   window.
//   3  root         SenderPipe::send() the finalized COMPACT stat tile to the
//                   group's cb_mcast_in (loopback multicast: src == dst, so the
//                   EXCLUDE-source path serves the root from its own copy).
//      non-root     ReceiverPipe::receive() into cb_mcast_in.
//                   Compute then un-permutes that one tile back into the
//                   BLOCK_ROWS column-shaped stat tiles pass B reads.
//   4  every core   drain THIS block's output (write_block) before starting the
//                   next round.  Not cosmetic: compute cannot finish block blk+1's
//                   pass A until it has drained block blk's pass B, so a writer
//                   that ran the whole combine first and the whole write-back
//                   second deadlocks the moment num_blocks exceeds the output CB's
//                   depth.
//
// It lives in the WRITER, not the reader, for two reasons: NoC1 is idle through
// pass A (so the combine handshake overlaps the reader's NoC0 x/gamma traffic),
// and cb_compact_handoff / cb_mcast_in then have exactly one dataflow kernel
// touching them — cb_sum_handoff, cb_row_final and cb_row_stat all stay
// compute-private, which is the CB-ownership rule the design calls out for this
// exact handoff.
//
// The gather landing address is `get_write_ptr(cb_partials_gathered)` computed
// LOCALLY on the sender: that CB is declared on every core of the program, so
// its L1 address is identical everywhere, and its ring holds exactly
// GATHER_SLOTS pages — one per sender, FLAT in BLOCK_ROWS since D27 — so a
// whole-round push returns the pointer to the base each round.  The host
// therefore never has to know a CB address.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"  // DataflowBuffer, for the boot zeroing below
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
// PERMANENT per-stage device-profiler instrumentation (never remove; free when
// the profiler is off -- see the header's durability contract).
// ---- TEMPORARY ABLATION SWITCH (/perf-measure cumulative peel) -------------
// Uncomment to strip the gather boot-zeroing payload.  Perf measurement only.
// #define RMS_ABLATE_GATHER_ZERO
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_output_tiles = 8;
constexpr uint32_t cb_output_sticks = 9;
constexpr uint32_t cb_sum_handoff = 10;
constexpr uint32_t cb_partials_gathered = 11;
constexpr uint32_t cb_stat_handoff = 12;
constexpr uint32_t cb_row_final = 13;
// Perf 3 / D27: on the COMPACT path (BLOCK_ROWS > 1) the gather source and the multicast
// landing move here, and cb_sum_handoff / cb_row_final become compute-private -- the
// permute consumes one and produces the other.  At BLOCK_ROWS == 1 the permute is the
// identity and is elided, so those two revert to being this kernel's endpoints.  The
// CB_GATHER_SRC / CB_MCAST_LAND aliases below are the single place that choice is made.
constexpr uint32_t cb_compact_handoff = 15;
constexpr uint32_t cb_mcast_in = 16;
constexpr uint32_t TILE_DIM = 32;
}  // namespace

void kernel_main() {
    // ---- compile-time knobs (all from rms_norm_program_descriptor.py) -----
    constexpr uint32_t IS_TILE = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_W_CHUNKS = get_compile_time_arg_val(3);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(4);
    constexpr uint32_t ELEM_BYTES = get_compile_time_arg_val(5);
    [[maybe_unused]] constexpr uint32_t R_RM = get_compile_time_arg_val(6);
    constexpr uint32_t W_ELEMS = get_compile_time_arg_val(7);
    constexpr uint32_t NATIVE_OUT = get_compile_time_arg_val(8);
    constexpr uint32_t COMBINE = get_compile_time_arg_val(9);
    constexpr uint32_t GATHER_SEM_ID = get_compile_time_arg_val(10);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(11);
    // Perf 2 (descriptor D22): the gather's slots per tile-row -- GROUP_SIZE rounded UP
    // TO EVEN.  The root's fused fold (compute kernel) walks a row's partials PAIRWISE in
    // one DEST window, halves p and p + GATHER_SLOTS/2, so an odd group needs one pad slot
    // to pair against.  Derived, not passed: it is a pure function of GROUP_SIZE and the
    // compute kernel derives it identically, so there is one definition of the layout in
    // each kernel and no CT arg can drift between them.  Equals GROUP_SIZE at every even
    // group (all of 8 / 28 / 32, and the focus shape's 8), so it is byte-identical there.
    constexpr uint32_t GATHER_SLOTS = GROUP_SIZE + GROUP_SIZE % 2;
    constexpr uint32_t OUT_SHARD_PAGES = get_compile_time_arg_val(12);
    // Refinement 2b: BAND == 1 means the output goes back stick-by-stick into this
    // core's own band -- straight into the resident output shard when
    // OUT_SHARD_ROW_BYTES != 0, otherwise through the accessor at the band's byte
    // offset (legal only for a stick-paged output: interleaved / height-sharded).
    constexpr uint32_t BAND = get_compile_time_arg_val(13);
    constexpr uint32_t OUT_SHARD_ROW_BYTES = get_compile_time_arg_val(14);
    // Refinement 4 (descriptor D13): faces per partial tile the IDENTITY-path gather ships.
    // 4 is the whole tile; 2 ships only the two faces that can hold a REDUCE_ROW column
    // vector, i.e. HALF the bytes.  Perf 3 / D27 confines it to the BLOCK_ROWS == 1 branch
    // (the compact branch must ship whole tiles) and MEASURES that it has to stay there --
    // see the gather below.
    constexpr uint32_t GATHER_FACES = get_compile_time_arg_val(15);
    constexpr auto mc = dataflow_kernel_lib::McastArgs</*CT=*/16, /*RT=*/10>();
    constexpr auto out_args = TensorAccessorArgs<mc.next_compile_time_args_offset()>();

    constexpr bool RM = (IS_TILE == 0);
    constexpr bool NATIVE = (NATIVE_OUT != 0);
    constexpr bool CROSS_CORE = (COMBINE != 0);
    constexpr bool BAND_OUT = (BAND != 0);
    constexpr bool BAND_OUT_LOCAL = (OUT_SHARD_ROW_BYTES != 0);
    static_assert(!BAND_OUT || IS_TILE == 0, "rms_norm: the BAND scheme is ROW_MAJOR-only");
    // The cross-core width combine used to be TILE-only: an RM shard that cuts the
    // width axis had no expressible per-core width slice.  The BAND scheme gives it
    // one (see the reader), and the combine itself is unchanged -- it sums per-row
    // partials elementwise and never cares where a band starts.
    static_assert(!CROSS_CORE || !RM || BAND_OUT, "rms_norm: an RM width combine needs the BAND scheme");
    static_assert(!CROSS_CORE || NUM_W_CHUNKS == 1, "rms_norm: a width-split core takes its slice in one chunk");
    // D27: one tile-row's stat per COLUMN of the compact tile, and a tile has 32 columns.
    static_assert(!CROSS_CORE || BLOCK_ROWS <= TILE_DIM, "rms_norm: a compact combine block is at most 32 tile-rows");
    // THE ONE CARVE-OUT (D27), and it is an IDENTITY rather than a benchmark boundary: at
    // BLOCK_ROWS == 1 a block is one tile-row, so permuting its stat "into column 0" is the
    // tile it already was.  Compute elides both matmuls there and the combine's endpoints
    // revert to cb_sum_handoff / cb_row_final.  MEASURED reason it must be elided rather
    // than left in as a uniform path: 0.76-0.80x whole-op on the four pinned WIDTH-shard
    // geometries (all BLOCK_ROWS == 1), because the permute pair adds an L1 round trip that
    // a single-round combine has no latency left to hide.  Full numbers at the `COMPACT`
    // definition in rms_norm_compute.cpp.  The GATHER ITSELF IS UNIFIED: both paths ship
    // ONE WHOLE TILE into page `my_slot` of a GATHER_SLOTS-page ring.
    constexpr bool COMPACT = CROSS_CORE && (BLOCK_ROWS > 1);
    constexpr uint32_t CB_GATHER_SRC = COMPACT ? cb_compact_handoff : cb_sum_handoff;
    constexpr uint32_t CB_MCAST_LAND = COMPACT ? cb_mcast_in : cb_row_final;
    constexpr uint32_t CHUNK_ROW_BYTES = WT_CHUNK * TILE_DIM * ELEM_BYTES;
    constexpr uint32_t LAST_CHUNK_ROW_BYTES = W_ELEMS * ELEM_BYTES - (NUM_W_CHUNKS - 1) * CHUNK_ROW_BYTES;

    // ---- runtime work assignment -----------------------------------------
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);  // this core's first tile-row
    const uint32_t num_rows = get_arg_val<uint32_t>(2);   // tile-rows owned by this core
    const uint32_t w_start = get_arg_val<uint32_t>(3);    // first width tile this core owns
    const uint32_t is_root = get_arg_val<uint32_t>(4);    // group root (multicast sender)
    const uint32_t my_slot = get_arg_val<uint32_t>(5);    // index within the width group
    // The ROW_MAJOR view of the same slice (mirrors the reader's args 6..9).
    const uint32_t stick_base = get_arg_val<uint32_t>(6);
    const uint32_t stick_count = get_arg_val<uint32_t>(7);
    const uint32_t w_off_elems = get_arg_val<uint32_t>(8);
    const uint32_t w_real_elems = get_arg_val<uint32_t>(9);

    // An INACTIVE core (see the reader): it exists only so the stat multicast
    // lands in a cb_row_final this program owns.  No shard, no work.
    if (num_rows == 0) {
        return;
    }

    const auto out_acc = TensorAccessor(out_args, out_addr);
    const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);

    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;

    // ---- BAND write-back: the reader's stage_band, mirrored ------------------
    // Same transaction granularity as the read half (one per tile-row when the band
    // fills its tile columns and the shard stride matches, one per stick
    // otherwise), so neither NoC half is the batched one.
    const uint32_t band_bytes = w_real_elems * ELEM_BYTES;
    const uint32_t band_off_bytes = w_off_elems * ELEM_BYTES;
    // The band sits at lane (w_off_elems % 32) of each untilized stick -- the
    // reader's GLOBAL TILE FRAME, mirrored.
    const uint32_t band_delta_bytes = (w_off_elems % TILE_DIM) * ELEM_BYTES;
    const bool band_contiguous = BAND_OUT_LOCAL && (band_delta_bytes == 0) && (band_bytes == CHUNK_ROW_BYTES) &&
                                 (OUT_SHARD_ROW_BYTES == CHUNK_ROW_BYTES);
    auto write_band = [&](uint32_t stick_start, uint32_t sticks) {
        for (uint32_t s = 0; s < sticks; s += TILE_DIM) {
            const uint32_t n = ((sticks - s) < TILE_DIM) ? (sticks - s) : TILE_DIM;
            cb_wait_front(cb_output_sticks, WT_CHUNK);
            const uint32_t src = get_read_ptr(cb_output_sticks) + band_delta_bytes;
            if (band_bytes != 0) {
                if (band_contiguous) {
                    const uint32_t dst = out_addr + (stick_start + s - stick_base) * OUT_SHARD_ROW_BYTES;
                    noc_async_write(src, get_noc_addr(dst), n * CHUNK_ROW_BYTES);
                } else {
                    for (uint32_t i = 0; i < n; ++i) {
                        const uint64_t dst =
                            BAND_OUT_LOCAL
                                ? get_noc_addr(out_addr + (stick_start + s + i - stick_base) * OUT_SHARD_ROW_BYTES)
                                : out_acc.get_noc_addr(stick_start + s + i, band_off_bytes);
                        noc_async_write(src + i * CHUNK_ROW_BYTES, dst, band_bytes);
                    }
                }
                noc_async_write_barrier();
            }
            cb_pop_front(cb_output_sticks, WT_CHUNK);
        }
    };

    // ---- one row-block of output ---------------------------------------------
    // A LAMBDA, not a second loop, because the cross-core combine below has to
    // interleave with it: the combine's round for block `blk+1` cannot start until
    // compute has finished block `blk`, and compute cannot finish block `blk` until
    // this writer has drained its output.  Running the whole combine first and the
    // whole write-back second deadlocks as soon as num_blocks exceeds the output
    // CB's depth -- which the TILE schemes never hit only because their shard fits
    // in ONE row-block, and which the ROW_MAJOR BAND scheme hits immediately (its
    // per-block gather CB is GROUP_SIZE fp32 tiles, so L1 caps BLOCK_ROWS low).
    auto write_block = [&](uint32_t blk) {
        MaybeDeviceZoneScope("writer_write");
        if constexpr (NATIVE) {
            return;  // zero-copy: compute packed into the shard; the pages ARE the tensor
        }
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
        const uint32_t first_tile_row = row_start + r0;

        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            if constexpr (RM) {
                const uint32_t stick_start = stick_base + r0 * TILE_DIM;
                uint32_t sticks = rows * TILE_DIM;
                if (r0 * TILE_DIM + sticks > stick_count) {
                    sticks = stick_count - r0 * TILE_DIM;  // short final tile-row
                }
                if constexpr (BAND_OUT) {
                    write_band(stick_start, sticks);
                } else {
                    const uint32_t row_bytes = (c + 1 == NUM_W_CHUNKS) ? LAST_CHUNK_ROW_BYTES : CHUNK_ROW_BYTES;
                    dataflow_kernel_lib::write_sticks_after_untilize<cb_output_sticks>(
                        out_acc, sticks, row_bytes, stick_start, /*byte_offset_within_page=*/c * CHUNK_ROW_BYTES);
                }
            } else {
                for (uint32_t r = 0; r < rows; ++r) {
                    // + w_start: this core's width slice under a cross-core width
                    // split (0 on the whole-row schemes).
                    const uint32_t tile_base = (first_tile_row + r) * WT + w_start + c * WT_CHUNK;
                    cb_wait_front(cb_output_tiles, WT_CHUNK);
                    uint32_t l1_addr = get_read_ptr(cb_output_tiles);
                    for (uint32_t w = 0; w < WT_CHUNK; ++w) {
                        const uint32_t wt = w_start + c * WT_CHUNK + w;
                        if (wt < WT) {  // a ragged width shard ends in pad tiles
                            noc_async_write_tile(tile_base + w, out_acc, l1_addr);
                        }
                        l1_addr += out_tile_bytes;
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_output_tiles, WT_CHUNK);
                }
            }
        }
    };

    // The combine's pipes and semaphore are built ONCE, above the loop (their
    // ctors are the documented handshake init) and only on a participating core.
    Noc noc;
    Semaphore<> gather_sem(CROSS_CORE ? GATHER_SEM_ID : 0);
    uint32_t arrivals = 0;

    if constexpr (CROSS_CORE) {
        const uint32_t stat_bytes = get_tile_size(cb_stat_handoff);
        // ---- THE COMPACT GATHER (Perf 3, descriptor D27) --------------------
        // The gather is the only per-GROUP_SIZE term in the combine: every member ships
        // into ONE root, so the root's L1 ingress carries (GROUP_SIZE - 1) transfers per
        // round while the stat multicast back carries ONE tile however big the group is.
        // It is therefore where a transfer lever has the fan-in multiplier on it.
        //
        // WHAT USED TO BE HERE (D16 + D13).  A partial is a REDUCE_ROW result and does not
        // fill its tile, so a sender shipped its BLOCK_ROWS column-shaped partials as
        // BLOCK_ROWS separate landings at page `r * GATHER_SLOTS + my_slot`, each cut down
        // to the two 16x16 faces that can carry a column vector -- i.e. TWO face-sized NoC
        // writes per tile-row per member.  At the 64-core BLOCK-shard geometry
        // (GROUP_SIZE 8, BLOCK_ROWS 8) that is 16 writes / 16 kB per member per round to
        // carry 8 rows x 32 fp32 = 1 kB of actual information: 16x byte amplification in
        // sub-face-sized chunks, and BLOCK_ROWS times the transaction count.
        //
        // WHAT IT IS NOW.  Compute has permuted the whole block into ONE tile whose
        // columns are its BLOCK_ROWS stats (D27, `member_pack` in rms_norm_compute.cpp),
        // so a member ships that ONE WHOLE TILE into page `my_slot`: one transaction,
        // 4 kB, flat in BLOCK_ROWS.  MEASURED (isolated bench
        // perf_experiments/compact_partial_transpose_r3, blackhole p150b 1350 MHz, at the
        // op's pinned config): `writer_gather_ship` on a member 1891 -> 1087 ns/round, the
        // root's `writer_mcast_send` 4133 -> 1147 and a member's `writer_mcast_recv`
        // 6577 -> 1395, whole combine 34772 -> 10994 ns (3.16x).
        //
        // ON THE COMPACT PATH IT CANNOT GO BACK TO A FACE SUBSET -- a correctness property,
        // not a preference.  The receiver's un-permute is a matmul, which sums 32 products
        // across the row, so ANY un-written column of a landing page becomes inf*0 = NaN in
        // column 0.  Shipping the whole tile out of a fully-defined `pack_tile` makes every
        // byte of every real landing page written by exactly one member per round --
        // defined by construction, which is also what leaves D26's face-zeroing with
        // nothing to zero and reduces the odd-GROUP_SIZE pad below to ONE page.
        //
        // AND ON THE IDENTITY PATH (BLOCK_ROWS == 1) D13's FACE-RUN GATHER STAYS, MEASURED.
        // There is no un-permute matmul there, so nothing requires the un-shipped faces to
        // be defined (D26's measurement: the garbage in faces 1/3 is carried and never
        // read), and the whole-tile ship DOUBLES the bytes into the root -- which is the one
        // place in this op with a GROUP_SIZE fan-in multiplier on it.  Shipping whole tiles
        // at BLOCK_ROWS == 1 was measured a whole-op REGRESSION, MONOTONE IN GROUP_SIZE,
        // which is that multiplier's own signature (one fresh-cache profiled run each):
        //     (1,1,32,1024)  8c   3724 -> 3916 ns   0.951x
        //     (1,1,32,2304)  9c   4527 -> 4621 ns   0.980x
        //     (1,1,32,5120) 32c   5406 -> 6290 ns   0.859x
        //     (1,1,32,7168) 28c   5724 -> 6304 ns   0.908x
        // So the two paths ship differently for a measured reason on BOTH sides: the compact
        // gather cannot be narrowed, and the identity gather must not be widened.  Above
        // BLOCK_ROWS == 1 the extra bytes are bought back many times over -- the compact
        // ship is ONE transaction for the whole block instead of 2 x BLOCK_ROWS.
        //
        // ONE definition of the member -> root transfer, used by the root for its own slot
        // (local) and by every member (remote).  `dst_noc` is the root's gather-CB BASE.
        static_assert(GATHER_FACES >= 2 && GATHER_FACES <= 4, "rms_norm: GATHER_FACES must be 2, 3 or 4");
        const uint32_t face_bytes = stat_bytes / 4;
        auto ship_partial = [&](uint32_t src, uint64_t dst_noc) {
            const uint32_t d_off = my_slot * stat_bytes;
            if constexpr (COMPACT || GATHER_FACES == 4) {
                noc_async_write(src, dst_noc + d_off, stat_bytes);
            } else if constexpr (GATHER_FACES == 3) {
                noc_async_write(src, dst_noc + d_off, 3 * face_bytes);
            } else {
                // Faces 0 and 2 -- the only pair that can carry a REDUCE_ROW column vector.
                noc_async_write(src, dst_noc + d_off, face_bytes);
                noc_async_write(src + 2 * face_bytes, dst_noc + d_off + 2 * face_bytes, face_bytes);
            }
        };
        // ---- THE ONE THING STILL ZEROED: the odd-GROUP_SIZE PAD PAGE -------------------
        // NO FACE ZEROING (Perf 3 / D26, MEASURED -- do not restore).  On the IDENTITY path
        // faces 1 and 3 of every landing page are still un-shipped and hold whatever was in
        // the root's L1.  Refinement 4 through Perf 2 zeroed them here; that work is DELETED,
        // and the safety argument is a MEASUREMENT rather than a claim.  D26's bench
        // (perf_experiments/gather_zero_elim) seeded those lanes with NINE catastrophic
        // patterns -- 1e30, -1e30, NaN, +-Inf, fp32 subnormals, a per-lane mix that makes the
        // fold evaluate Inf + (-Inf) = NaN, and a stale-L1 lookalike -- ran the op's REAL
        // fold + finalize + pass B consumer at GROUP_SIZE 4/8/9/28/32, and got output
        // BIT-IDENTICAL (torch.equal) to the zeroed run every time, with a bit-identical stat
        // column 0.  The control that makes that a proof: the packed stat tile's columns
        // 16..31 came back 100% non-finite for the NaN/Inf seeds -- the garbage DOES enter
        // DEST and IS multicast, it is CARRIED and never READ, because the only reader of a
        // stat tile is pass B's column broadcast (column 0, faces 0 and 2, both shipped).
        // Cost removed: 2462 ns (7.1%) of the focus shape's wall.  On the COMPACT path the
        // question does not even arise -- the page is shipped whole.  If a future consumer
        // ever reads a gathered or stat tile WHOLE, this deletion must be revisited.
        //
        // What survives is a different question with a different answer.
        //
        // A PAD SLOT (D22 rounds GATHER_SLOTS up to even so the root's pairwise walk has an
        // even count) is never written by ANY member -- a sender lands at page `my_slot`
        // with `my_slot < GROUP_SIZE <= GATHER_SLOTS - 1` -- and it is folded WHOLE into the
        // group sum, so it must be an exact +0.0.  MEASURED catastrophic without it: pad =
        // 1e30 gives rel-RMS 1.00 at pcc 0.999672 (the uniform-scale error pcc alone does
        // not see, which is exactly why this op's nets bound rms and not just pcc), and
        // pad = NaN gives rel-RMS 1.00.  Under D27 it matters MORE, not less: the
        // un-permute matmul sums 32 products, so a non-finite pad column poisons column 0.
        // Zeroing the whole page rather than a face subset is also measured: this stage pays
        // per API CALL, not per byte (Noc::async_write_zeros sets the read state up once and
        // chunks at MEM_ZEROS_SIZE = 512 B), so two 1024 B calls lose to one 4096 B call by
        // 10-11% (1959 vs 1781 ns at GROUP_SIZE 9 / BLOCK_ROWS 8).
        //
        // D27 shrinks the stage from GATHER_SLOTS * BLOCK_ROWS pages to GATHER_SLOTS, i.e.
        // to exactly ONE pad page per round-independent ring -- and it is a boot-time
        // one-shot because the ring is reused in place every round.
        //
        // Zeroing the whole CB is still WRONG for the reason Refinement 4 measured (pcc
        // 0.87-0.99): it would wipe members that had already landed.  Only the root reads
        // this CB, so only the root pays; at even GROUP_SIZE (4/8/28/32, including every
        // even live profile) the stage does literally nothing.
        //
        // ABLATION PEEL RECIPE (kept working against the compact path): uncomment
        // RMS_ABLATE_GATHER_ZERO at the head of this file to strip the payload while the
        // `is_root` predicate and the zone stay; diff the profiled zones against the
        // unablated run.  Pair it with RMS_ABLATE_ROOT_SUM in rms_norm_compute.cpp to peel
        // the root chain cumulatively.  Both are perf-only -- the op is WRONG with either on
        // at odd GROUP_SIZE.
        constexpr bool ZERO_PAD = (GATHER_SLOTS != GROUP_SIZE);
        if constexpr (ZERO_PAD) {
#if defined(RMS_ABLATE_GATHER_ZERO)
            if (false) {
#else
            if (is_root != 0) {
#endif
                MaybeDeviceZoneScope("writer_gather_zero");
                DataflowBuffer gather_dfb(cb_partials_gathered);
                const uint32_t pages = gather_dfb.get_total_size_bytes() / stat_bytes;
                for (uint32_t p = 0; p < pages; ++p) {
                    if (p % GATHER_SLOTS >= GROUP_SIZE) {  // a pad slot: zero it whole
                        noc.async_write_zeros(gather_dfb, stat_bytes, {.offset_bytes = p * stat_bytes});
                    }
                }
                noc.write_zeros_l1_barrier();
            }
        }
        if (is_root != 0) {
            auto sender = mc.sender(noc);
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                // 1. the root's own COMPACT partial goes into its own slot of its own
                //    gather CB.  D27: ONE page, so the window is GATHER_SLOTS whatever
                //    `rows` is -- a RAGGED last block needs no special case, and both
                //    kernels derive the identical window, which is what keeps a remote
                //    sender's locally-computed landing address equal to the root's.
                {
                    MaybeDeviceZoneScope("writer_gather_ship");
                    cb_wait_front(CB_GATHER_SRC, 1);
                    cb_reserve_back(cb_partials_gathered, GATHER_SLOTS);
                    ship_partial(get_read_ptr(CB_GATHER_SRC), get_noc_addr(get_write_ptr(cb_partials_gathered)));
                    noc_async_write_barrier();
                }
                // NO self-signal here.  Semaphore::up(value) is a NON-ATOMIC local
                // read-modify-write (noc_semaphore.h: "multiple cores incrementing
                // simultaneously may lead to lost updates"), so a local bump on the
                // root would race the members' remote atomic incs and silently drop
                // one -- a hang in whichever group lost the race.  The root's own
                // slot is written synchronously above, so it only ever waits for the
                // OTHER GROUP_SIZE - 1 members.
                cb_pop_front(CB_GATHER_SRC, 1);

                // 2. publish the gathered round once every member has landed.
                {
                    MaybeDeviceZoneScope("writer_gather_wait");
                    arrivals += GROUP_SIZE - 1;
                    gather_sem.wait_min(arrivals);
                    cb_push_back(cb_partials_gathered, GATHER_SLOTS);
                }

                // 3. multicast the finalized stat back to the whole group.
                //
                // The root places its OWN copy first, then broadcasts in place
                // (src == dst => EXCLUDE-source).  Doing it this way makes the two
                // host emitters behave identically: Mcast1D's per-row sender rect
                // EXCLUDES the sender (mcast_host.hpp sender_rect_), while Mcast2D's
                // rect contains it -- an in-place send takes the same EXCLUDE path
                // in both, so the root is never served twice and never skipped.
                {
                    MaybeDeviceZoneScope("writer_mcast_send");
                    // D27: ONE compact tile carries the whole round's stats, so the
                    // broadcast is 1 page instead of BLOCK_ROWS -- measured 4133 -> 1147
                    // ns/round on the root and 6577 -> 1395 on a member.  It lands in
                    // cb_mcast_in (not cb_row_final): compute un-permutes it into the
                    // `rows` column-shaped tiles pass B reads, so cb_row_final is now
                    // compute-private.
                    cb_wait_front(cb_stat_handoff, 1);
                    cb_reserve_back(CB_MCAST_LAND, 1);
                    const uint32_t stat_dst = get_write_ptr(CB_MCAST_LAND);
                    noc_async_write(get_read_ptr(cb_stat_handoff), get_noc_addr(stat_dst), stat_bytes);
                    noc_async_write_barrier();
                    // Perf 2 (D24): PUBLISH THE ROOT'S OWN COPY BEFORE THE BROADCAST.
                    // The root's un-permute (and behind it its pass B) blocks on this push,
                    // so pushing after the send made the root wait out the whole multicast
                    // to the other GROUP_SIZE-1 cores even though its own copy of the stat
                    // had been in L1 since before the send started.  Legal because `send()`
                    // and the un-permute are both READERS of this page -- the send never
                    // writes it -- and cb_mcast_in is CB_COMBINE_FLAT_DEPTH (== 2) pages
                    // deep, so the next round's reserve cannot reach the page the
                    // (already-returned) send read.
                    //
                    // MEASURED on the root core (perf_experiments/combine_pipeline_depth):
                    // `compute_scale` 13575 -> 10932 ns, i.e. -2643 ns of the root's pass B
                    // spent waiting out its own multicast.  Whole-op it is worth 1.006x
                    // ALONE but 1.037x on top of the compute-side pipeline (which exposes
                    // the root's pass B as the next thing on the critical path).
                    cb_push_back(CB_MCAST_LAND, 1);
                    if constexpr (mc.active) {
                        sender.send(stat_dst, stat_dst, stat_bytes);
                    }
                    cb_pop_front(cb_stat_handoff, 1);
                }

                // 4. drain THIS block's output before the next combine round, so
                //    compute is never blocked on a full output CB (see write_block).
                write_block(blk);
            }
        } else {
            auto receiver = mc.receiver(noc);
            const uint32_t root_x = mc.sender_x();
            const uint32_t root_y = mc.sender_y();
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                // 1. ship this core's COMPACT partial to the root's slot, then signal.
                //    ONE whole-tile write per round (D27), whatever BLOCK_ROWS is.  The
                //    landing address is `get_write_ptr(cb_partials_gathered)` computed
                //    LOCALLY: a member never reserves/pushes/pops that CB, so its pointer
                //    is always the base, and the root's ring is exactly GATHER_SLOTS pages
                //    so its pointer returns to the base every round too.
                {
                    MaybeDeviceZoneScope("writer_gather_ship");
                    cb_wait_front(CB_GATHER_SRC, 1);
                    ship_partial(
                        get_read_ptr(CB_GATHER_SRC), get_noc_addr(root_x, root_y, get_write_ptr(cb_partials_gathered)));
                    noc_async_write_barrier();  // data before signal
                    gather_sem.up(noc, root_x, root_y, 1);
                    cb_pop_front(CB_GATHER_SRC, 1);
                }

                // 3. reserve the landing slot FIRST: receive()'s ack means "free".
                //    This member's ship for round blk+1 is ordered AFTER this receive, and
                //    the root only sends once its fold has popped the gather ring -- which
                //    is the happens-before chain that lets the ring be ONE round deep.
                {
                    MaybeDeviceZoneScope("writer_mcast_recv");
                    cb_reserve_back(CB_MCAST_LAND, 1);
                    receiver.receive();
                    cb_push_back(CB_MCAST_LAND, 1);
                }

                // 4. same interleave as the root's.
                write_block(blk);
            }
        }
    } else {
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            write_block(blk);
        }
    }

    if constexpr (NATIVE) {
        // Zero-copy output: compute packed into the shard itself (write_block was a
        // no-op).  Take the completion barrier and leave the pages pushed -- they
        // ARE the tensor.
        cb_wait_front(cb_output_tiles, num_rows * WT_CHUNK);
    }
}
