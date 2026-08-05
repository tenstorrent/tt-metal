// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
// ISOLATED BAKE-OFF COPY -- perf_experiments/root_rotation.  NOT the op.
// ============================================================================
// VERBATIM copy of ttnn/ttnn/operations/rms_norm/kernels/rms_norm_writer.cpp plus
// an `RMS_ROT_VARIANT` bitmask:
//
//   variant 0 (ROTATE clear)  the op TODAY -- one FIXED root per group does every
//                             round's gather + fold + multicast.  The honest baseline.
//   bit 1 ROTATE              the root duty ROTATES PER ROW-BLOCK: round blk's root is
//                             the group member at slot (blk + rot_phase) % GROUP_SIZE.
//                             Every core is therefore sometimes the multicast sender
//                             and gather destination, and sometimes a receiver + gather
//                             source.
//   bit 2 ZDEFER              under ROTATE, a core whose FIRST root round is not round
//                             0 does the gather zeroing AFTER its round-0 ship (where
//                             its BRISC would be parked in the mcast wait) instead of
//                             at boot, where it would delay that ship and with it the
//                             whole group's round 0.
//   bit 4 DIAG                under ROTATE, each width group's rotation is phase-shifted
//                             by its grid row, so the groups' roots are never all in one
//                             column on the same round.
//   bit 8 NOZERO              ABLATION ONLY -- strip the gather zeroing payload so its
//                             cost is separable.  Never a candidate.
//
// WHY THE SWITCH IS A PREPROCESSOR `#if` AND NOT `if constexpr`: with `if constexpr`
// the baseline measured 40.3 us against the op's own 34.4 us on the focus geometry --
// the discarded rotating branch still inflated the BRISC binary enough to cost 5.9 us
// of wall.  A `#if` makes variant 0's translation unit BYTE-IDENTICAL to the op's
// writer (same mcast RT base, same inline zero block, same loop pair), and it
// re-measured at 34.4 us.  A baseline that is not the op is a strawman, so this
// spelling is load-bearing, not stylistic.
//
// The rotating multicast is NOT hand-rolled: `McastConfig.rotating_sender` /
// `McastArgs<CT, RT, SPAN>` / `SenderPipe<..., ROTATING_SENDER>` already exist in the
// helper library for exactly this topology (host/mcast_host.hpp + mcast_pipe.hpp), so
// the rotating wire is the helper's and the only thing this file spells is the
// per-round role predicate.
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
//   1  every core   write its BLOCK_ROWS raw partial tiles (cb_sum_handoff) into
//                   its own slot of the ROOT's cb_partials_gathered, then
//                   remote-inc the root's arrival semaphore.
//   2  root         once arrivals reach (blk+1) * GROUP_SIZE, publish the
//                   gathered block so compute can sum + finalize it.
//   3  root         SenderPipe::send() the finalized stat tiles to the group's
//                   cb_row_final (loopback multicast: src != dst, so the root
//                   gets its own copy too).
//      non-root     ReceiverPipe::receive() into cb_row_final.
//   4  every core   drain THIS block's output (write_block) before starting the
//                   next round.  Not cosmetic: compute cannot finish block blk+1's
//                   pass A until it has drained block blk's pass B, so a writer
//                   that ran the whole combine first and the whole write-back
//                   second deadlocks the moment num_blocks exceeds the output CB's
//                   depth.
//
// It lives in the WRITER, not the reader, for two reasons: NoC1 is idle through
// pass A (so the combine handshake overlaps the reader's NoC0 x/gamma traffic),
// and cb_sum_handoff / cb_row_final then have exactly one dataflow kernel
// touching them — cb_row_stat stays compute-private, which is the CB-ownership
// rule the design calls out for this exact handoff.
//
// The gather landing address is `get_write_ptr(cb_partials_gathered)` computed
// LOCALLY on the sender: that CB is declared on every core of the program, so
// its L1 address is identical everywhere, and its ring holds exactly
// GROUP_SIZE * BLOCK_ROWS pages so a whole-block push returns the pointer to the
// base each round.  The host therefore never has to know a CB address.

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

// ---- the bake-off switch (see the header) ----------------------------------
#ifndef RMS_ROT_VARIANT
#define RMS_ROT_VARIANT 0
#endif
#define RMS_ROT (RMS_ROT_VARIANT & 1)
#define RMS_ZDEFER (RMS_ROT_VARIANT & 2)
#define RMS_NOZERO (RMS_ROT_VARIANT & 8)
// ATTRIBUTION variant (bit 16, requires ROTATE): the whole ROTATING MACHINERY -- the
// rotating mcast wire, the full-line EXCLUDE-source rect, both pipes on every core, the
// per-round sender-coord indexing, the bigger BRISC binary -- with the PLACEMENT held
// STILL at slot 0.  It answers "how much of rotation's cost is the mechanism and how
// much is moving the root?", which no fixed/rotating pair can separate.
#define RMS_STILL (RMS_ROT_VARIANT & 16)

// The op's gather-zeroing payload, VERBATIM, as a MACRO so the rotating variants can
// place it at two different points in the writer without a function call.
//
// IT WAS A LAMBDA FIRST, AND THAT WAS A MEASUREMENT BUG: `auto z = [&](){...}` called
// from two sites is not inlined, and the by-reference captures turn the 64-page loop's
// operands into memory reads.  `writer_gather_zero` measured 7837 ns inline vs 13664 ns
// through the lambda -- +5827 ns, which is 2/3 of what first looked like "the cost of
// rotation".  Whatever this idea's verdict is, it must not be an artifact of how the
// bench spells a helper.
#define RMS_GATHER_ZERO()                                                                                   \
    do {                                                                                                    \
        if constexpr (NEEDS_ZERO) {                                                                         \
            MaybeDeviceZoneScope("writer_gather_zero");                                                     \
            DataflowBuffer gather_dfb(cb_partials_gathered);                                                \
            const uint32_t pages = gather_dfb.get_total_size_bytes() / stat_bytes;                          \
            for (uint32_t p = 0; p < pages; ++p) {                                                          \
                const uint32_t base = p * stat_bytes;                                                       \
                if constexpr (ZERO_PAD) {                                                                   \
                    if (p % GATHER_SLOTS >= GROUP_SIZE) { /* a pad slot: zero it whole */                   \
                        noc.async_write_zeros(gather_dfb, stat_bytes, {.offset_bytes = base});              \
                        continue;                                                                           \
                    }                                                                                       \
                }                                                                                           \
                if constexpr (GATHER_FACES == 2) { /* faces 1 and 3 unshipped */                            \
                    noc.async_write_zeros(gather_dfb, face_bytes, {.offset_bytes = base + face_bytes});     \
                }                                                                                           \
                if constexpr (GATHER_FACES < 4) {                                                           \
                    noc.async_write_zeros(gather_dfb, face_bytes, {.offset_bytes = base + 3 * face_bytes}); \
                }                                                                                           \
            }                                                                                               \
            noc.write_zeros_l1_barrier();                                                                   \
        }                                                                                                   \
    } while (0)

namespace {
constexpr uint32_t cb_output_tiles = 8;
constexpr uint32_t cb_output_sticks = 9;
constexpr uint32_t cb_sum_handoff = 10;
constexpr uint32_t cb_partials_gathered = 11;
constexpr uint32_t cb_stat_handoff = 12;
constexpr uint32_t cb_row_final = 13;
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
    // Refinement 4 (descriptor D13): faces per partial tile the GATHER ships.  4 is
    // the whole tile (Refinement 2/3's behaviour); 2 ships only the two faces that
    // can hold a REDUCE_ROW column vector.  See COMPACT_GATHER below.
    constexpr uint32_t GATHER_FACES = get_compile_time_arg_val(15);
#if RMS_ROT
    // BENCH WIRE (rotating variants only): rot_phase takes runtime arg 10, so the mcast
    // block starts at 11.  SPAN > 0 selects the helper's ROTATING wire (a full-line rect
    // + one sender coord pair per round) AND the rotating-sender pipe behaviour (send()
    // resets this core's own data-ready flag after the broadcast, so its next RECEIVER
    // turn does not return on its own stale VALID).  SPAN is GROUP_SIZE: the rotation
    // cycle is the group, and on the PACKED single-group topology (Mcast2D over a
    // bounding box that also holds INACTIVE cores) the host's coord list is
    // rect-row-major, whose first GROUP_SIZE entries are exactly the active cores in
    // slot order -- so reading only GROUP_SIZE pairs is correct and the extra trailing
    // words the host emits are simply unread.
    constexpr auto mc = dataflow_kernel_lib::McastArgs</*CT=*/16, /*RT=*/11, GROUP_SIZE>();
#else
    constexpr auto mc = dataflow_kernel_lib::McastArgs</*CT=*/16, /*RT=*/10>();
#endif
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
    constexpr uint32_t CHUNK_ROW_BYTES = WT_CHUNK * TILE_DIM * ELEM_BYTES;
    constexpr uint32_t LAST_CHUNK_ROW_BYTES = W_ELEMS * ELEM_BYTES - (NUM_W_CHUNKS - 1) * CHUNK_ROW_BYTES;

    // ---- runtime work assignment -----------------------------------------
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);  // this core's first tile-row
    const uint32_t num_rows = get_arg_val<uint32_t>(2);   // tile-rows owned by this core
    const uint32_t w_start = get_arg_val<uint32_t>(3);    // first width tile this core owns
    // BENCH: unused in the ROTATE build (the role is a per-block predicate there), so
    // tag it rather than delete it -- the baseline path below must stay the op's text.
    [[maybe_unused]] const uint32_t is_root = get_arg_val<uint32_t>(4);  // group root (multicast sender)
    const uint32_t my_slot = get_arg_val<uint32_t>(5);                   // index within the width group
    // The ROW_MAJOR view of the same slice (mirrors the reader's args 6..9).
    const uint32_t stick_base = get_arg_val<uint32_t>(6);
    const uint32_t stick_count = get_arg_val<uint32_t>(7);
    const uint32_t w_off_elems = get_arg_val<uint32_t>(8);
    const uint32_t w_real_elems = get_arg_val<uint32_t>(9);
#if RMS_ROT
    // BENCH: rotation phase for THIS core's group.  LINE-UNIFORM by construction (the
    // host emits the same value to every core of a group) -- if it were not, a group's
    // cores would disagree about who the round's root is and the gather semaphore would
    // never reach its arrival count: a HANG, not a slowdown.
    const uint32_t rot_phase = get_arg_val<uint32_t>(10);
#endif

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
        // ---- COMPACT GATHER (Refinement 4, descriptor D13) ------------------
        // The gather is the only per-GROUP_SIZE term in the combine: every member
        // ships its partial into ONE root, so the root's L1 ingress carries
        // (GROUP_SIZE - 1) * BLOCK_ROWS tiles per round while the stat multicast
        // back carries BLOCK_ROWS tiles TOTAL however big the group is.  Shrinking
        // the gather transfer is therefore the byte lever with the fan-in
        // multiplier on it, and the mcast is deliberately left whole-tile.
        //
        // A partial is a REDUCE_ROW result, so it does not fill its tile: a 32x32
        // tile is stored as 2x2 faces of 16x16 (1024 B each at offsets 0 / 1024 /
        // 2048 / 3072), and the reduce writes only the parts of it that pass B's
        // mul<BroadcastDim::Col> can read back.  GATHER_FACES is how many LEADING
        // faces a member ships -- 4 is the whole tile, 3 drops the trailing face and
        // is a third fewer bytes, all in ONE transaction per tile either way.
        //
        // Whatever the gather does not write stays whatever was in the root's L1, so
        // the number is only lowerable as far as the faces the datapath actually
        // reads; see D13 for the measurement that fixed it.
        //   4  the whole tile.
        //   3  faces 0,1,2 contiguously -- one transfer per tile, 3/4 of the bytes.
        //   2  faces 0 and 2 only (the pair that can hold a column vector) -- two
        //      face-sized transfers per tile, HALF the bytes.
        static_assert(GATHER_FACES >= 2 && GATHER_FACES <= 4, "rms_norm: GATHER_FACES must be 2, 3 or 4");
        const uint32_t face_bytes = stat_bytes / 4;
        // ROW-MAJOR GATHER LAYOUT (Perf 1, descriptor D16).  A sender's partial for
        // row r lands at PAGE `r * GATHER_SLOTS + my_slot` of the root's gather CB, i.e.
        // the group's GROUP_SIZE partials for ONE row are CONTIGUOUS.  That is what lets
        // the root fold a whole row inside ONE DEST window (compute kernel: pairwise
        // add_tiles with acc_to_dest, then the finalize, then one pack -- Perf 2 / D22)
        // instead of GROUP_SIZE separate helper calls.  Perf 2 widens the stride from
        // GROUP_SIZE to GATHER_SLOTS so the pairwise walk has an even count to halve.  The Phase-0 layout was
        // slot-major
        // (`my_slot * rows + r`), which put a row's partials on a stride of `rows` --
        // a gapped window no chain walk can express.
        //
        // The cost is transfer COUNT, not bytes: a sender now issues one transfer per
        // row instead of one per block (GATHER_FACES == 4 loses its single contiguous
        // run).  The gather is tiny (a `rows`-tile column vector per sender) and rides
        // NoC1, which is idle through pass A; the root-side serial fold it unlocks was
        // measured at 6.6 us of an 11.2 us decode profile.
        //
        // ONE definition of the member -> root partial transfer, used by the root for
        // its own slot (local) and by every member (remote).  `dst_noc` is the root's
        // gather-CB BASE (slot offset is applied per row here, not by the caller).
        auto ship_partial = [&](uint32_t src, uint64_t dst_noc, uint32_t rows) {
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t s_off = r * stat_bytes;
                const uint32_t d_off = (r * GATHER_SLOTS + my_slot) * stat_bytes;
                if constexpr (GATHER_FACES == 4) {
                    noc_async_write(src + s_off, dst_noc + d_off, stat_bytes);
                } else if constexpr (GATHER_FACES == 3) {
                    noc_async_write(src + s_off, dst_noc + d_off, 3 * face_bytes);
                } else {
                    noc_async_write(src + s_off, dst_noc + d_off, face_bytes);
                    noc_async_write(src + s_off + 2 * face_bytes, dst_noc + d_off + 2 * face_bytes, face_bytes);
                }
            }
        };
        // Boot: make the faces the gather never writes DEFINED, so no undefined L1
        // ever reaches the root's elementwise sum / rsqrt.  Zeroing exactly the
        // UNSHIPPED faces (and nothing else) is what makes this race-free: a member's
        // partial can land at any time, and it only ever touches faces the root
        // leaves alone -- which is why zeroing the whole CB here does NOT work (it
        // wipes members that already arrived; measured as pcc 0.87-0.99 with a large
        // rms across every combine cell).  Only the root reads this CB, so only the
        // root pays.
        // A PAD slot (odd GROUP_SIZE only, D22) is never written by any member, so the
        // root's fold would otherwise add whatever L1 garbage was there.  Zeroing it WHOLE
        // is race-free by exactly the argument below: a sender lands at
        // `r * GATHER_SLOTS + my_slot` with `my_slot < GROUP_SIZE <= GATHER_SLOTS - 1`, so
        // no member ever touches a pad page, and the pad contributes an exact +0.0.
        constexpr bool ZERO_PAD = (GATHER_SLOTS != GROUP_SIZE);

#if RMS_ROT
        // ================= ROOT ROTATION (the candidate) ======================
        // Round `blk`'s root is the group member at slot (blk + rot_phase) % GROUP_SIZE.
        // Both faces of the channel are built on every active core: it is the multicast
        // SENDER and gather destination on its own rounds, a RECEIVER and gather source
        // on the others.  Both ctors are local and NoC-free, so the cores that never root
        // pay nothing for holding them.
        //
        // The zeroing payload is the op's, verbatim, lifted into a lambda so ZDEFER can
        // place it somewhere other than boot.  It only has to happen before THIS core's
        // own fold READS the gather CB, and it is race-free against member landings by
        // construction (it writes exactly the faces no member ever writes -- see the
        // COMPACT GATHER comment above), so it may be deferred arbitrarily late.
        constexpr bool NEEDS_ZERO = (GATHER_FACES < 4 || ZERO_PAD) && !RMS_NOZERO;

        auto sender = mc.sender(noc);
        auto receiver = mc.receiver(noc);

        // THE FIRST ROUND THIS CORE ROOTS, and whether it ever does.  Roots are the slots
        // {(blk + rot_phase) mod GROUP_SIZE : blk < num_blocks}, so this core's first turn
        // is at blk == (my_slot - rot_phase) mod GROUP_SIZE.  When num_blocks < GROUP_SIZE
        // only the first num_blocks of them ever root; the rest skip the zeroing entirely,
        // which is why rotation does not simply multiply the boot-zero cost by GROUP_SIZE.
#if RMS_STILL
        const uint32_t first_root = 0;
        const bool will_root = (my_slot == 0);
#else
        const uint32_t first_root = (my_slot + GROUP_SIZE - (rot_phase % GROUP_SIZE)) % GROUP_SIZE;
        const bool will_root = first_root < num_blocks;
#endif
        if (will_root && (!RMS_ZDEFER || first_root == 0)) {
            RMS_GATHER_ZERO();
        }

        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            const uint32_t r0 = blk * BLOCK_ROWS;
            const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
            // The round index doubles as the receiver's index into the helper's per-round
            // sender-coord list (which is in LINE / slot order), so the two can never
            // disagree about who is broadcasting.
            const uint32_t round = RMS_STILL ? 0u : ((blk + rot_phase) % GROUP_SIZE);

            if (my_slot == round) {
                // 1. this round's root puts its own partial in its own slot, LOCALLY.
                {
                    MaybeDeviceZoneScope("writer_gather_ship");
                    cb_wait_front(cb_sum_handoff, rows);
                    cb_reserve_back(cb_partials_gathered, GATHER_SLOTS * rows);
                    ship_partial(get_read_ptr(cb_sum_handoff), get_noc_addr(get_write_ptr(cb_partials_gathered)), rows);
                    noc_async_write_barrier();
                }
                // STILL no local self-signal, and the rule matters MORE here, not less:
                // under rotation every core is sometimes the local writer of this cell and
                // sometimes a remote incrementer of another core's, so a local
                // Semaphore::up (a NON-atomic read-modify-write) would race the members'
                // remote atomic incs and drop one -- a hang.  `arrivals` accumulates only
                // on the rounds this core roots, which is exactly when its own cell is
                // incremented.
                cb_pop_front(cb_sum_handoff, rows);
                {
                    MaybeDeviceZoneScope("writer_gather_wait");
                    arrivals += GROUP_SIZE - 1;
                    gather_sem.wait_min(arrivals);
                    cb_push_back(cb_partials_gathered, GATHER_SLOTS * rows);
                }
                {
                    MaybeDeviceZoneScope("writer_mcast_send");
                    cb_wait_front(cb_stat_handoff, rows);
                    cb_reserve_back(cb_row_final, rows);
                    const uint32_t stat_dst = get_write_ptr(cb_row_final);
                    noc_async_write(get_read_ptr(cb_stat_handoff), get_noc_addr(stat_dst), rows * stat_bytes);
                    noc_async_write_barrier();
                    cb_push_back(cb_row_final, rows);  // D24: own copy BEFORE the broadcast
                    if constexpr (mc.active) {
                        // src == dst => EXCLUDE-source, and the ROTATING rect is the WHOLE
                        // line (it must be -- the sender moves), so the sender is always
                        // interior and always takes that same EXCLUDE path.  The fixed op
                        // relies on the identical property.
                        sender.send(stat_dst, stat_dst, rows * stat_bytes);
                    }
                    cb_pop_front(cb_stat_handoff, rows);
                }
            } else {
                const uint32_t root_x = mc.sender_x(round);
                const uint32_t root_y = mc.sender_y(round);
                {
                    MaybeDeviceZoneScope("writer_gather_ship");
                    cb_wait_front(cb_sum_handoff, rows);
                    // The landing base is THIS core's own gather-CB write pointer, legal for
                    // exactly the reason the op records: the CB is declared on every core so
                    // its L1 address is identical, and a whole-block push returns the ring
                    // to its base.  Under rotation a core pushes it only on its own root
                    // rounds, so every core's pointer sits at the base when a round starts.
                    ship_partial(
                        get_read_ptr(cb_sum_handoff),
                        get_noc_addr(root_x, root_y, get_write_ptr(cb_partials_gathered)),
                        rows);
                    noc_async_write_barrier();  // data before signal
                    gather_sem.up(noc, root_x, root_y, 1);
                    cb_pop_front(cb_sum_handoff, rows);
                }
                // ZDEFER's landing spot: this core's round-0 ship is done and signalled, so
                // the group's round 0 is no longer waiting on it, and the zeroing now runs
                // where this BRISC would have been parked in the mcast wait.
                if (RMS_ZDEFER && blk == 0 && will_root && first_root != 0) {
                    RMS_GATHER_ZERO();
                }
                {
                    MaybeDeviceZoneScope("writer_mcast_recv");
                    cb_reserve_back(cb_row_final, rows);
                    receiver.receive(round);
                    cb_push_back(cb_row_final, rows);
                }
            }
            write_block(blk);
        }
#else
        // ================= THE OP TODAY (the baseline) ========================
        // Everything from here to the matching #endif is the op's writer, unedited
        // except for the RMS_NOZERO ablation hook replacing the op's own
        // RMS_ABLATE_GATHER_ZERO one.
        if constexpr (GATHER_FACES < 4 || ZERO_PAD) {
#if RMS_NOZERO
            if (false) {
#else
            if (is_root != 0) {
#endif
                MaybeDeviceZoneScope("writer_gather_zero");
                DataflowBuffer gather_dfb(cb_partials_gathered);
                const uint32_t pages = gather_dfb.get_total_size_bytes() / stat_bytes;
                for (uint32_t p = 0; p < pages; ++p) {
                    const uint32_t base = p * stat_bytes;
                    if constexpr (ZERO_PAD) {
                        if (p % GATHER_SLOTS >= GROUP_SIZE) {  // a pad slot: zero it whole
                            noc.async_write_zeros(gather_dfb, stat_bytes, {.offset_bytes = base});
                            continue;
                        }
                    }
                    if constexpr (GATHER_FACES == 2) {  // faces 1 and 3 unshipped
                        noc.async_write_zeros(gather_dfb, face_bytes, {.offset_bytes = base + face_bytes});
                    }
                    if constexpr (GATHER_FACES < 4) {
                        noc.async_write_zeros(gather_dfb, face_bytes, {.offset_bytes = base + 3 * face_bytes});
                    }
                }
                noc.write_zeros_l1_barrier();
            }
        }
        if (is_root != 0) {
            auto sender = mc.sender(noc);
            for (uint32_t blk = 0; blk < num_blocks; ++blk) {
                const uint32_t r0 = blk * BLOCK_ROWS;
                const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;

                // 1. the root's own partial goes into slot 0 of its own gather CB.
                {
                    MaybeDeviceZoneScope("writer_gather_ship");
                    cb_wait_front(cb_sum_handoff, rows);
                    cb_reserve_back(cb_partials_gathered, GATHER_SLOTS * rows);
                    ship_partial(get_read_ptr(cb_sum_handoff), get_noc_addr(get_write_ptr(cb_partials_gathered)), rows);
                    noc_async_write_barrier();
                }
                // NO self-signal here.  Semaphore::up(value) is a NON-ATOMIC local
                // read-modify-write (noc_semaphore.h: "multiple cores incrementing
                // simultaneously may lead to lost updates"), so a local bump on the
                // root would race the members' remote atomic incs and silently drop
                // one -- a hang in whichever group lost the race.  The root's own
                // slot is written synchronously above, so it only ever waits for the
                // OTHER GROUP_SIZE - 1 members.
                cb_pop_front(cb_sum_handoff, rows);

                // 2. publish the gathered block once every member has landed.
                {
                    MaybeDeviceZoneScope("writer_gather_wait");
                    arrivals += GROUP_SIZE - 1;
                    gather_sem.wait_min(arrivals);
                    cb_push_back(cb_partials_gathered, GATHER_SLOTS * rows);
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
                    cb_wait_front(cb_stat_handoff, rows);
                    cb_reserve_back(cb_row_final, rows);
                    const uint32_t stat_dst = get_write_ptr(cb_row_final);
                    noc_async_write(get_read_ptr(cb_stat_handoff), get_noc_addr(stat_dst), rows * stat_bytes);
                    noc_async_write_barrier();
                    // Perf 2 (D24): PUBLISH THE ROOT'S OWN COPY BEFORE THE BROADCAST.
                    // The root's pass B blocks on this push, so pushing after the send made
                    // the root wait out the whole multicast to the other GROUP_SIZE-1 cores
                    // even though its own copy of the stat had been in L1 since before the
                    // send started.  Legal because `send()` and pass B are both READERS of
                    // these pages -- the send never writes them -- and cb_row_final is
                    // CB_ROW_STAT_DEPTH (== 2) row-blocks deep, so the next round's reserve
                    // cannot reach the half the (already-returned) send read.
                    //
                    // MEASURED on the root core (perf_experiments/combine_pipeline_depth):
                    // `compute_scale` 13575 -> 10932 ns, i.e. -2643 ns of the root's pass B
                    // spent waiting out its own multicast.  Whole-op it is worth 1.006x
                    // ALONE but 1.037x on top of the compute-side pipeline (which exposes
                    // the root's pass B as the next thing on the critical path).
                    cb_push_back(cb_row_final, rows);
                    if constexpr (mc.active) {
                        sender.send(stat_dst, stat_dst, rows * stat_bytes);
                    }
                    cb_pop_front(cb_stat_handoff, rows);
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
                const uint32_t r0 = blk * BLOCK_ROWS;
                const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;

                // 1. ship this core's raw partial to the root's slot, then signal.
                {
                    MaybeDeviceZoneScope("writer_gather_ship");
                    cb_wait_front(cb_sum_handoff, rows);
                    ship_partial(
                        get_read_ptr(cb_sum_handoff),
                        get_noc_addr(root_x, root_y, get_write_ptr(cb_partials_gathered)),
                        rows);
                    noc_async_write_barrier();  // data before signal
                    gather_sem.up(noc, root_x, root_y, 1);
                    cb_pop_front(cb_sum_handoff, rows);
                }

                // 3. reserve the landing slot FIRST: receive()'s ack means "free".
                {
                    MaybeDeviceZoneScope("writer_mcast_recv");
                    cb_reserve_back(cb_row_final, rows);
                    receiver.receive();
                    cb_push_back(cb_row_final, rows);
                }

                // 4. same interleave as the root's.
                write_block(blk);
            }
        }
#endif  // RMS_ROT
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
