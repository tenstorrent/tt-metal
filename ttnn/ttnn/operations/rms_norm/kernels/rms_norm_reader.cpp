// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm reader (NCRISC / NoC0).
//
// Feeds, per row-block owned by this core:
//   TILE input      -> cb_input_tiles   (BLOCK_HT * chunk_wt tiles, one barrier per block)
//   ROW_MAJOR input -> cb_rm_in         (chunk_wt tile-pages per tile-row of 32 sticks)
//   gamma           -> cb_gamma_tiles   (resident in Regime A, per-chunk in Regime B)
//   reduce scaler   -> cb_reduce_scaler (once per core)
//
// ---------------------------------------------------------------------------
// CB-WRAP INVARIANT (why every access here is a fixed size)
// ---------------------------------------------------------------------------
// A multi-page cb_reserve_back / cb_wait_front followed by a CONTIGUOUS N-page
// access is only legal when the CB's page count is a multiple of N and the fifo
// pointer is N-aligned; otherwise the access runs off the end of the CB into the
// neighbouring one (silent, deterministic corruption).  Two invariants keep that
// true here, both enforced by the host plan:
//   * the W-chunk DIVIDES Wt_core, so there is no short trailing chunk, and
//   * every row-block is exactly BLOCK_HT tile-rows.  For the final row-block of
//     the tensor the phantom tile-rows are CLAMPED to the last valid one (a
//     cheap re-read that keeps the data finite); the writer discards them.
//
// ---------------------------------------------------------------------------
// HELPER SUBSTITUTIONS (documented before the body, per the implementer rules)
// ---------------------------------------------------------------------------
// 1. The ROW_MAJOR stick read does NOT go through
//    dataflow_kernel_lib::read_sticks_for_tilize().  That helper reads
//    `row_bytes` per stick and leaves the (padded_row_bytes - row_bytes) tail of
//    every L1 row UNINITIALISED.  Regime A's `maskless_w` predicate is valid
//    *only* because the reader zero-fills that tail (op_design.md "Reader
//    obligations on the RM path" #1), and in Regime B a stale `inf` there would
//    become NaN through the masked scaler (risk R3).  The helper also owns its
//    own block loop, which cannot express this op's (row-block x W-chunk)
//    iteration order or the phantom-row clamp.  The body below is otherwise
//    structurally identical to the helper's TILE-granularity path: one
//    cb_reserve_back + N reads + ONE noc_async_read_barrier + one cb_push_back
//    per tile-row (lever B7).
//
// 2. ROW_MAJOR gamma is staged through `cb_gamma_rm` + a compute-side tilize()
//    (op_design.md "Gamma ingest"), but the staging CB is CHUNKED at
//    GAMMA_INGEST_BLOCK tiles instead of holding the full padded width.  A
//    full-width staging buffer would cost Wt * tile_bytes (1 MB at W = 16384)
//    purely to hold tile rows 1..31 that the downstream `BroadcastDim::Row`
//    multiply never reads.  GAMMA_INGEST_BLOCK divides every ingest count the
//    kernel uses, so tilize<GAMMA_INGEST_BLOCK> never over-produces gamma tiles.
//    (Placing tile row 0 directly with two per-face reads was tried first and is
//    NOT legal HERE: the second face read starts 32 B into a stick, so its L1 and
//    DRAM 64 B RESIDUES differ (32 vs 0) and Blackhole's sanitizer rule fires.
//    Contrast the TILE-gamma span in #3, whose runs all start 64 B-aligned on both
//    ends - the rule is a residue match, not a size alignment.)
//
// 3. There is NO helper for "read only the broadcast row of a tile page".  The
//    TILE-layout gamma read below is a raw `noc_async_read` of a computed byte
//    SPAN instead of `noc_async_read_tile` / any kernel_lib page reader, because
//    every reader in the library is page-granular: it takes a page id and moves
//    `page_size` bytes.  A `BroadcastDim::Row` consumer needs 1/32 of that (bf16:
//    544 of 2048 B), and expressing it requires knowing the FACE layout of a
//    tile, which no dataflow helper exposes.  This is a CAPABILITY gap, not an
//    ergonomic bypass - a helper such as
//        read_bcast_row_tiles(acc, first_page, n, cb, TileFormat)
//    that owns the face-layout arithmetic (and the block-format exponent header)
//    would let this site, and every other Row/Col-broadcast operand read in the
//    library, drop the hand geometry.  Until it exists the arithmetic lives here,
//    fenced by static_asserts so an unhandled tile format is a BUILD error.
//
// 4. The W-SPLIT COMBINE's N->1 GATHER leg (W_SPLIT=1, `kernel_main`'s combine
//    block) is raw `noc_async_write` + `noc_semaphore_inc`/`_wait` and NOT
//    `mcast_pipe`.  CAPABILITY, not ergonomics:
//      * `SenderPipe::send(src_l1, dst_l1, size)` (mcast_pipe.hpp:189-197) is a
//        1->N MULTICAST of ONE source region to an `McastRect` of destinations.
//        The gather is the OPPOSITE direction - an N->1 unicast fan-in where each
//        of the G senders lands in a DIFFERENT slot of ONE core's CB.  No
//        constructor in that header expresses it.
//      * `ReceiverPipe`'s NUM_SENDERS (mcast_pipe.hpp:242, 255-261) governs
//        multi-sender SIGNALLING - which stored coord pair to ack on round `r` -
//        not a multi-source DATA fan-in; it keeps coords, never landing slots.
//    The 1->N leg DOES use the helper (`SenderPipe`/`ReceiverPipe` below) and must
//    keep doing so.  The isolated bake-off ran a hand-rolled arm for that leg too -
//    GROUP_SIZE-1 unicasts + GROUP_SIZE-1 semaphore incs, same handshake shape -
//    and the helper WON at every group size measured, because it collapses those
//    G-1 point-to-point writes into one multicast transaction.  So the split here
//    is not a style preference: raw for the direction no helper expresses, helper
//    for the direction it does.  Do not "restore" the gather to a helper - there
//    is none - and do not hand-roll the broadcast.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

using dataflow_kernel_lib::ACK_EQUALS_FANOUT;
using dataflow_kernel_lib::McastRect;
using dataflow_kernel_lib::ReceiverPipe;
using dataflow_kernel_lib::SenderPipe;

namespace {

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma_tiles = 1;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_sumsq = 4;
constexpr uint32_t cb_rm_in = 8;
constexpr uint32_t cb_sumsq_acc = 10;
constexpr uint32_t cb_gamma_rm = 11;
constexpr uint32_t cb_partial_gather = 12;
constexpr uint32_t cb_sumsq_bcast = 13;

// --- shared geometry compile-time args (identical prefix in all 3 kernels) ---
constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(0);
constexpr uint32_t REGIME_A = get_compile_time_arg_val(1);
constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(2);
constexpr uint32_t GAMMA_IS_ROW_MAJOR = get_compile_time_arg_val(3);
constexpr uint32_t Wt_core = get_compile_time_arg_val(4);
constexpr uint32_t W_PARTIAL = get_compile_time_arg_val(5);
constexpr uint32_t BLOCK_HT = get_compile_time_arg_val(6);
constexpr uint32_t WT_REDUCE_BLOCK = get_compile_time_arg_val(7);
constexpr uint32_t WT_SCALE_BLOCK = get_compile_time_arg_val(8);
constexpr uint32_t Rt = get_compile_time_arg_val(9);
constexpr uint32_t NUM_ROWS = get_compile_time_arg_val(10);
constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(11);
constexpr uint32_t ELEM_SIZE = get_compile_time_arg_val(12);
constexpr uint32_t GAMMA_ELEM_SIZE = get_compile_time_arg_val(13);
constexpr uint32_t GAMMA_ROW_BYTES = get_compile_time_arg_val(14);
constexpr uint32_t DEST_BLOCK_CT = get_compile_time_arg_val(15);
constexpr uint32_t GAMMA_TILE_BYTES = get_compile_time_arg_val(16);
constexpr uint32_t IN_TILE_BYTES = get_compile_time_arg_val(17);
constexpr uint32_t GAMMA_INGEST_BLOCK = get_compile_time_arg_val(18);
// Lever B7: 1 = one noc barrier per block (applied), 0 = one per transaction.
constexpr uint32_t BARRIER_PER_BLOCK = get_compile_time_arg_val(19);
// /perf-measure ablation: keep every CB op and barrier, issue no NoC transfer.
constexpr uint32_t SKIP_DM_PAYLOAD = get_compile_time_arg_val(20);
// Lever B5/B6: 1 = one whole-page transaction per tile (applied), 0 = two half-page ones.
constexpr uint32_t COALESCE = get_compile_time_arg_val(21);
// Regime B reduce datapath (see rms_norm_compute.cpp).  It selects which FORM of
// the non-tile-aligned partial the compute side consumes, so the reader has to
// emit the matching tile - see the scaler block in kernel_main().
constexpr uint32_t REDUCE_VIA_ADD = get_compile_time_arg_val(22);
// --- W-split work distribution (blocking_plan._choose_group_size) ------------
// W_SPLIT == 0 is the row-parallel plan: every branch below compiles out and this
// kernel is byte-identical to the pre-split one.
constexpr uint32_t W_SPLIT = get_compile_time_arg_val(23);
constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(24);
constexpr uint32_t WT_TOTAL = get_compile_time_arg_val(25);
constexpr uint32_t ACC_TILE_BYTES = get_compile_time_arg_val(26);
constexpr uint32_t SEM_DATA_READY = get_compile_time_arg_val(27);
constexpr uint32_t SEM_CONSUMER_READY = get_compile_time_arg_val(28);
constexpr uint32_t SEM_GATHER = get_compile_time_arg_val(29);

// The DRAM tile-row stride.  Under a W split a core owns Wt_core of WT_TOTAL
// columns, so the stride is the FULL row width, not the per-core slice.
constexpr uint32_t ROW_STRIDE = W_SPLIT ? WT_TOTAL : Wt_core;

// =====================================================================
// TILE-layout gamma: ROW-0 SPAN geometry (the only part of a gamma tile read)
// =====================================================================
// gamma feeds a `BroadcastDim::Row` multiply (rms_norm_compute.cpp
// `scale_chunk`), which reads tile ROW 0 of each gamma tile and nothing else -
// rows 1..31 of a gamma tile are pure tile padding that the datapath never
// touches.  The reader therefore fetches ONE transaction per gamma tile covering
// [0, end-of-row-0) of the page instead of the whole page: same transaction
// COUNT as a whole-page read (this shape is transaction-bound, not byte-bound -
// the two-run "exactly row 0" variant issues 2x the transactions and measured
// SLOWER), 3.8x fewer bytes on bf16.  Rows 1..31 of the L1 page are left
// UNWRITTEN; a NaN-poison probe (fill the CB with NaN bit patterns, then run)
// confirms the Row broadcast never reads them.
//
// RUN GEOMETRY, per supported gamma dtype (page bytes -> span bytes):
//   bfloat16    2048 -> 544     fp32     4096 -> 1088     bfloat8_b 1088 -> 336
// A non-block tile is four 16x16 faces stored contiguously: face 0 is rows 0-15
// / cols 0-15 at page offset 0, face 1 is rows 0-15 / cols 16-31 at offset
// 256*elem.  Tile row 0 is thus two runs of 16*elem bytes at offsets 0 and
// 256*elem, and the span is [0, 256*elem + 16*elem).
// A bfloat8_b tile is 1088 B: a 64 B exponent header (one byte per face-row, 4
// faces x 16 rows) then four 256 B mantissa faces.  Row 0 of face f needs
// exponent byte 16*f and mantissa run [64 + 256*f, +16), so run 1 ends at
// 64 + 256 + 16 = 336.  The whole 64 B header is inside the span, which is why
// the per-face exponents row 0 needs are always present.
//
// NoC LEGALITY (Blackhole).  The sanitizer rule for a DRAM read is a RESIDUE
// MATCH, not "the transfer size is a multiple of the alignment":
//     (worker_addr & alignment_mask) == (noc_addr & alignment_mask)
// with alignment_mask = NOC_DRAM_READ_ALIGNMENT_BYTES - 1 = 63
// (tt_metal/hw/inc/internal/debug/sanitize.h:602, mask set at :530).  Every
// DRAM gamma page starts 64 B-aligned (aligned_page_size = align(tile_bytes,64)
// = 2048 / 4096 / 1088, all multiples of 64) and the cb_gamma_tiles page stride
// is the same tile size, so src and dst residues are both 0 on every supported
// gamma dtype.  Only the LENGTH is sub-alignment, which the rule does not
// constrain.  That is exactly why this span is legal while the ROW_MAJOR-gamma
// two-face read (substitution #2 above) is NOT: there the second face read
// starts 32 B into a stick, so the residues are 32 vs 0 and the rule fires.
constexpr bool GAMMA_IS_BLOCK = (GAMMA_ELEM_SIZE == 0);
constexpr uint32_t G_FACE_ROW = GAMMA_IS_BLOCK ? 16u : 16u * GAMMA_ELEM_SIZE;
constexpr uint32_t G_EXP_BYTES = GAMMA_IS_BLOCK ? 64u : 0u;
constexpr uint32_t G_FACE = GAMMA_IS_BLOCK ? 256u : 256u * GAMMA_ELEM_SIZE;
// Span end = start of face 1 + row 0 of face 1.
constexpr uint32_t G_SPAN_LEN = G_EXP_BYTES + G_FACE + G_FACE_ROW;
static_assert((G_EXP_BYTES + G_FACE) % 64 == 0, "gamma row-0 span: face-1 base must be 64 B-aligned");
static_assert(!HAS_GAMMA || G_SPAN_LEN <= GAMMA_TILE_BYTES, "gamma row-0 span cannot exceed the page");
// The block-format geometry above is bfloat8_b-SPECIFIC (64 B header, 256 B
// faces).  bfloat4_b packs a 32 B header and 128 B faces, so these constants
// would silently fetch the WRONG bytes.  Fail the BUILD, not the numerics, if a
// new block gamma dtype ever reaches here.
static_assert(
    !HAS_GAMMA || !GAMMA_IS_BLOCK || GAMMA_TILE_BYTES == 1088,
    "gamma row-0 span: block-format geometry is bfloat8_b-only (1088 B tile)");
// PRECONDITION, not assertable here (TILE_DIM is declared below): the run
// geometry assumes the standard 32x32 tile of four 16x16 faces, which is what
// this op's host plan hard-codes everywhere (TILE_DIM = 32).  A non-32x32 gamma
// tile would trip the two static_asserts above via GAMMA_TILE_BYTES.

// Lever B5/B6 off-arm: the tile page split into TWO transfers.  The split point
// must stay NoC-alignment-legal on every dtype - Blackhole's DRAM alignment is
// 64 B, and a bfloat8_b tile is 1088 B, whose midpoint (544) is NOT 64 B-aligned.
// Rounding the first half DOWN to a 64 B multiple keeps both offsets legal and
// still covers the whole page (1088 -> 512 + 576).
constexpr uint32_t SPLIT_FIRST = (IN_TILE_BYTES / 2) & ~static_cast<uint32_t>(63);
constexpr uint32_t SPLIT_SECOND = IN_TILE_BYTES - SPLIT_FIRST;

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t NUM_REDUCE_CHUNKS = Wt_core / WT_REDUCE_BLOCK;
constexpr uint32_t NUM_SCALE_CHUNKS = Wt_core / WT_SCALE_BLOCK;
constexpr uint32_t LAST_RT = Rt - 1;

constexpr auto input_args = TensorAccessorArgs<30>();
[[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

FORCE_INLINE uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Zero `n` bytes of L1 starting at `addr`.  Only ever called on the padded tail
// of a row-major stick (< 32 elements), so the byte loop is bounded and cheap.
FORCE_INLINE void zero_l1(uint32_t addr, uint32_t n) {
    volatile tt_l1_ptr uint8_t* p = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(addr);
    for (uint32_t i = 0; i < n; ++i) {
        p[i] = 0;
    }
}

}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_row_block = get_arg_val<uint32_t>(2);
    const uint32_t num_row_blocks_here = get_arg_val<uint32_t>(3);
    // W-split: this core's column base inside the full row (tiles), and its role.
    // Both are 0 on the row-parallel plan, where every W_SPLIT branch is gone.
    const uint32_t W_OFFSET = get_arg_val<uint32_t>(4);
    const uint32_t IS_ROOT = get_arg_val<uint32_t>(5);

    const auto in_acc = TensorAccessor(input_args, src_addr);

    // ---- The SUM scaler is exactly 1.0.  1/W is applied later in fp32 by the
    //      compute chain, so no scalar is ever quantised to bf16 (risk R2).
    //      Both regimes need it: Regime A finalises `sum_of_squares`' tile
    //      accumulator with a within-tile REDUCE_ROW, Regime B reduces every
    //      W-chunk.  Only Regime B needs the PARTIAL tile - it is the only one
    //      that reduces the raw last W-tile, whose pad columns must be zeroed
    //      (risk R1).  In Regime A the accumulator's 32 columns are all
    //      meaningful (the pad only ever lives in the last W-tile, and the RM
    //      reader zero-fills it), so a full scaler is the correct one.
    // PERMANENT per-stage instrumentation (kernel_lib/perf_instrumentation.hpp).
    // Every NoC region is split into `_reserve` (back-pressure from the
    // consumer), `_issue` (RISC-serial transaction issue) and `_barrier` (the
    // real NoC wait): a barrier at ~0 with a hot issue loop and a hot barrier
    // with a cheap issue loop want opposite fixes.
    {
        MaybeDeviceZoneScope("rd_scaler");
        if constexpr (!REGIME_A && W_PARTIAL > 0) {
            // The two reduce datapaths consume DIFFERENT forms of the partial, in
            // different tile layouts, so the tile the reader emits at index 1 is
            // chosen by the same REDUCE_VIA_ADD knob the compute side reads:
            //   ReduceTile       -> a PARTIAL SCALER tile (matmul-with-ones layout);
            //                       compute passes ReducePartialScaler::last_tile_at(1).
            //   AccumulateViaAdd -> a 0/1 MASK tile in row-0 broadcast layout, which
            //                       the masked accumulating broadcast-mul folds into
            //                       the last tile; compute passes partial_mask(W_PARTIAL, 1).
            // Passing the ReduceTile form to AccumulateViaAdd is silent, catastrophic
            // data corruption, not a compile error: valid_reduce_dim_elements stays 0,
            // the datapath reads "tile-aligned" and NEVER masks, so the poisoned tile
            // padding enters the sum of squares (measured rms ~1.0 on every
            // w_non_aligned pad-poison case).
            if constexpr (REDUCE_VIA_ADD) {
                dataflow_kernel_lib::
                    prepare_reduce_scaler<cb_reduce_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                        1.0f);
                dataflow_kernel_lib::prepare_reduce_mask<cb_reduce_scaler, ckernel::ReduceDim::REDUCE_ROW>(W_PARTIAL);
            } else {
                dataflow_kernel_lib::prepare_partial_reduce_scalers<
                    cb_reduce_scaler,
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    W_PARTIAL>(1.0f);
            }
        } else {
            dataflow_kernel_lib::
                prepare_reduce_scaler<cb_reduce_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
        }
    }  // rd_scaler

    // ---- gamma ingest -------------------------------------------------------
    // Places gamma tiles [w0, w0 + n) into cb_gamma_tiles (TILE gamma), or
    // stages them in cb_gamma_rm for the compute-side tilize (ROW_MAJOR gamma).
    auto fill_gamma = [&](uint32_t w0, uint32_t n) {
        if constexpr (HAS_GAMMA) {
            const auto g_acc = TensorAccessor(gamma_args, gamma_addr);
            if constexpr (GAMMA_IS_ROW_MAJOR) {
                constexpr uint32_t group_bytes = GAMMA_INGEST_BLOCK * TILE_DIM * GAMMA_ELEM_SIZE;
                for (uint32_t o = 0; o < n; o += GAMMA_INGEST_BLOCK) {
                    const uint32_t byte_off = (W_OFFSET + w0 + o) * TILE_DIM * GAMMA_ELEM_SIZE;
                    {
                        MaybeDeviceZoneScope("rd_gamma_reserve");
                        cb_reserve_back(cb_gamma_rm, GAMMA_INGEST_BLOCK);
                    }
                    const uint32_t addr = get_write_ptr(cb_gamma_rm);
                    if (byte_off < GAMMA_ROW_BYTES) {
                        {
                            MaybeDeviceZoneScope("rd_gamma_issue");
                            noc_async_read(
                                g_acc.get_noc_addr(0, byte_off), addr, umin(group_bytes, GAMMA_ROW_BYTES - byte_off));
                        }
                        {
                            MaybeDeviceZoneScope("rd_gamma_barrier");
                            noc_async_read_barrier();
                        }
                    }
                    cb_push_back(cb_gamma_rm, GAMMA_INGEST_BLOCK);
                }
            } else {
                {
                    MaybeDeviceZoneScope("rd_gamma_reserve");
                    cb_reserve_back(cb_gamma_tiles, n);
                }
                uint32_t addr = get_write_ptr(cb_gamma_tiles);
                {
                    MaybeDeviceZoneScope("rd_gamma_issue");
                    for (uint32_t i = 0; i < n; ++i) {
                        // ROW-0 SPAN, not the whole page: the consumer is a
                        // BroadcastDim::Row multiply, so only tile row 0 of this
                        // page is ever read.  One transaction, G_SPAN_LEN bytes;
                        // see the "TILE-layout gamma: ROW-0 SPAN geometry" block
                        // above for the per-dtype run geometry and the Blackhole
                        // residue-match legality argument.
                        noc_async_read(g_acc.get_noc_addr(W_OFFSET + w0 + i), addr, G_SPAN_LEN);
                        addr += GAMMA_TILE_BYTES;
                    }
                }
                {
                    MaybeDeviceZoneScope("rd_gamma_barrier");
                    noc_async_read_barrier();
                }
                cb_push_back(cb_gamma_tiles, n);
            }
        }
    };

    // ---- TILE input: one full BLOCK_HT x nw row-block chunk per call ---------
    auto read_tiles = [&](uint32_t rt0, uint32_t w0, uint32_t nw) {
        const uint32_t n = BLOCK_HT * nw;
        {
            MaybeDeviceZoneScope("rd_in_reserve");
            cb_reserve_back(cb_input_tiles, n);
        }
        uint32_t addr = get_write_ptr(cb_input_tiles);
        {
            MaybeDeviceZoneScope("rd_in_issue");
            for (uint32_t r = 0; r < BLOCK_HT; ++r) {
                const uint32_t row_base = umin(rt0 + r, LAST_RT) * ROW_STRIDE + W_OFFSET + w0;
                for (uint32_t w = 0; w < nw; ++w) {
                    if constexpr (!SKIP_DM_PAYLOAD) {
                        if constexpr (COALESCE) {
                            noc_async_read_tile(row_base + w, in_acc, addr);
                        } else {  // lever B5/B6 off-arm: two aligned partial-page transactions
                            noc_async_read(in_acc.get_noc_addr(row_base + w), addr, SPLIT_FIRST);
                            noc_async_read(
                                in_acc.get_noc_addr(row_base + w, SPLIT_FIRST), addr + SPLIT_FIRST, SPLIT_SECOND);
                        }
                    }
                    if constexpr (!BARRIER_PER_BLOCK) {
                        noc_async_read_barrier();  // lever B7 off-arm
                    }
                    addr += IN_TILE_BYTES;
                }
            }
        }  // rd_in_issue
        {
            MaybeDeviceZoneScope("rd_in_barrier");
            noc_async_read_barrier();
        }
        cb_push_back(cb_input_tiles, n);
    };

    // ---- ROW_MAJOR input: one tile-row (32 sticks) of a W-chunk per call ----
    auto read_sticks = [&](uint32_t rt, uint32_t w0, uint32_t nw) {
        const uint32_t row0 = umin(rt, LAST_RT) * TILE_DIM;
        const uint32_t nrows = umin(TILE_DIM, NUM_ROWS - row0);
        const uint32_t byte_off = (W_OFFSET + w0) * TILE_DIM * ELEM_SIZE;
        const uint32_t padded = nw * TILE_DIM * ELEM_SIZE;
        const uint32_t chunk_bytes = umin(padded, ROW_BYTES - byte_off);

        {
            MaybeDeviceZoneScope("rd_in_reserve");
            cb_reserve_back(cb_rm_in, nw);
        }
        const uint32_t base = get_write_ptr(cb_rm_in);
        uint32_t dst = base;
        {
            MaybeDeviceZoneScope("rd_in_issue");
            for (uint32_t r = 0; r < nrows; ++r) {
                if constexpr (!SKIP_DM_PAYLOAD) {
                    noc_async_read(in_acc.get_noc_addr(row0 + r, byte_off), dst, chunk_bytes);
                }
                if constexpr (!BARRIER_PER_BLOCK) {
                    noc_async_read_barrier();  // lever B7 off-arm
                }
                dst += padded;
            }
        }  // rd_in_issue
        {
            MaybeDeviceZoneScope("rd_in_barrier");
            noc_async_read_barrier();
        }
        // Zero the pad tail of every valid stick so tilize never promotes
        // uninitialised L1 into the reduction.  H-padding rows need no fill:
        // the reduction is per-row and the writer never emits a pad row.
        if (chunk_bytes < padded) {
            MaybeDeviceZoneScope("rd_zero_pad");
            uint32_t tail = base + chunk_bytes;
            for (uint32_t r = 0; r < nrows; ++r) {
                zero_l1(tail, padded - chunk_bytes);
                tail += padded;
            }
        }
        cb_push_back(cb_rm_in, nw);
    };

    auto read_input_chunk = [&](uint32_t rt0, uint32_t w0, uint32_t nw) {
        if constexpr (IS_ROW_MAJOR) {
            for (uint32_t r = 0; r < BLOCK_HT; ++r) {
                read_sticks(rt0 + r, w0, nw);
            }
        } else {
            read_tiles(rt0, w0, nw);
        }
    };

    // Regime A holds the whole per-core width of gamma resident for the whole
    // kernel: filled exactly once, never popped.  That is what makes the gamma
    // read cost 1x per core rather than 1x per row-block.
    if constexpr (REGIME_A) {
        fill_gamma(0, Wt_core);
    }

    // =====================================================================
    // W-SPLIT: the cross-core combine over the DEPENDENT axis.
    // =====================================================================
    // Per row-block, after this core has read its own Wt_core-wide slice and the
    // compute thread has folded it into ONE partial sum-of-squares accumulator
    // tile per tile-row (cb_sumsq_acc):
    //   step 2  N->1 GATHER   every core writes its BLOCK_HT partial tiles into
    //                         slot `w_index` of the group ROOT's cb_partial_gather
    //                         and increments the root's gather semaphore.  RAW NoC
    //                         - see justification #3 in the header.
    //   step 3  ROOT SUM      the root's compute thread sums the GROUP_SIZE
    //                         partials and collapses within-tile in ONE
    //                         reduce<..., AccumulateViaAdd> (rms_norm_compute.cpp).
    //   step 4  1->N BCAST    `SenderPipe::send` / `ReceiverPipe::receive`, loopback
    //                         ON so the root lands its own copy in the same call.
    //                         PRE_HANDSHAKE gates each broadcast on the receivers
    //                         having drained the previous row-block's tile, which is
    //                         also what keeps the NEXT gather from racing this one.
    // Held in its own `if constexpr` block because the two pipes must be
    // constructed ONCE outside the row-block loop, and constructing them at all is
    // illegal on the row-parallel plan (which creates no semaphores, so
    // ReceiverPipe's ctor `data_ready_.set(INVALID)` would write an L1 word this
    // program does not own).
    if constexpr (W_SPLIT) {
        Noc noc;
        const uint32_t root_vx = get_arg_val<uint32_t>(6);
        const uint32_t root_vy = get_arg_val<uint32_t>(7);
        const uint32_t sender_coords[2] = {root_vx, root_vy};

        // Both combine CBs hold EXACTLY one generation (blocking_plan asserts it),
        // so their fifo pointer is always the CB base -> the address read here is
        // the SAME L1 address on every core of the group.  That identity is what
        // lets a non-root core address the root's landing slot with no runtime
        // address table.
        const uint32_t gather_base = get_write_ptr(cb_partial_gather);
        const uint32_t sumsq_base = get_write_ptr(cb_sumsq);
        constexpr uint32_t PARTIAL_BYTES = BLOCK_HT * ACC_TILE_BYTES;
        const uint32_t w_index = W_OFFSET / Wt_core;

        Semaphore<> gather_sem(SEM_GATHER);
        SenderPipe<noc_index, SEM_DATA_READY, true, SEM_CONSUMER_READY> root_pipe(
            noc,
            McastRect<>{
                get_arg_val<uint32_t>(8),
                get_arg_val<uint32_t>(9),
                get_arg_val<uint32_t>(10),
                get_arg_val<uint32_t>(11)},
            ACK_EQUALS_FANOUT);
        ReceiverPipe<SEM_DATA_READY, true, SEM_CONSUMER_READY> leaf_pipe(noc, sender_coords);

        for (uint32_t b = 0; b < num_row_blocks_here; ++b) {
            const uint32_t rt0 = (start_row_block + b) * BLOCK_HT;
            read_input_chunk(rt0, 0, Wt_core);

            // ---- step 2: N->1 gather (raw NoC; header justification #3) -------
            // `cb_gather_partial_wait` is this core STARVED on its own compute
            // thread; `cb_gather_write` is the RISC-serial issue + the real NoC
            // barrier.  Split because a hot wait and a hot write want opposite fixes.
            {
                MaybeDeviceZoneScope("cb_gather_partial_wait");
                cb_wait_front(cb_sumsq_acc, BLOCK_HT);
            }
            {
                MaybeDeviceZoneScope("cb_gather_write");
                const uint32_t src = get_read_ptr(cb_sumsq_acc);
                for (uint32_t r = 0; r < BLOCK_HT; ++r) {
                    // Interleaved slot layout [ht][core] so the root's combine reduce
                    // reads a contiguous (BLOCK_HT x GROUP_SIZE) tile block.
                    const uint32_t slot = r * GROUP_SIZE + w_index;
                    noc_async_write(
                        src + r * ACC_TILE_BYTES,
                        get_noc_addr(root_vx, root_vy, gather_base + slot * ACC_TILE_BYTES),
                        ACC_TILE_BYTES);
                }
                noc_async_write_barrier();  // the DATA must be visible before the COUNT
                gather_sem.up(noc, root_vx, root_vy, 1);
                cb_pop_front(cb_sumsq_acc, BLOCK_HT);
            }

            if (IS_ROOT) {
                {
                    MaybeDeviceZoneScope("cb_gather_wait");
                    cb_reserve_back(cb_partial_gather, GROUP_SIZE * BLOCK_HT);
                    gather_sem.wait(GROUP_SIZE);
                    gather_sem.set(0);  // reset BEFORE the broadcast releases the leaves
                    cb_push_back(cb_partial_gather, GROUP_SIZE * BLOCK_HT);
                }
                {
                    MaybeDeviceZoneScope("mcast_src_wait");
                    cb_wait_front(cb_sumsq_bcast, BLOCK_HT);
                    cb_reserve_back(cb_sumsq, BLOCK_HT);
                }
                {
                    MaybeDeviceZoneScope("mcast_send");
                    root_pipe.send(get_read_ptr(cb_sumsq_bcast), sumsq_base, PARTIAL_BYTES);
                }
                cb_push_back(cb_sumsq, BLOCK_HT);
                cb_pop_front(cb_sumsq_bcast, BLOCK_HT);
            } else {
                cb_reserve_back(cb_sumsq, BLOCK_HT);
                {
                    MaybeDeviceZoneScope("mcast_recv");
                    leaf_pipe.receive();
                }
                cb_push_back(cb_sumsq, BLOCK_HT);
            }
        }
        return;  // the W-split arm owns its whole row-block loop
    }

    for (uint32_t b = 0; b < num_row_blocks_here; ++b) {
        const uint32_t rt0 = (start_row_block + b) * BLOCK_HT;

        if constexpr (REGIME_A) {
            read_input_chunk(rt0, 0, Wt_core);
        } else {
            // pass A - reduction
            for (uint32_t c = 0; c < NUM_REDUCE_CHUNKS; ++c) {
                read_input_chunk(rt0, c * WT_REDUCE_BLOCK, WT_REDUCE_BLOCK);
            }
            // pass B - scale (re-read of x, plus this chunk's gamma slice).
            // gamma FIRST: the compute kernel consumes it in the same order and
            // the staging CB is depth-1, so reversing the order deadlocks.
            for (uint32_t c = 0; c < NUM_SCALE_CHUNKS; ++c) {
                fill_gamma(c * WT_SCALE_BLOCK, WT_SCALE_BLOCK);
                read_input_chunk(rt0, c * WT_SCALE_BLOCK, WT_SCALE_BLOCK);
            }
        }
    }
}
