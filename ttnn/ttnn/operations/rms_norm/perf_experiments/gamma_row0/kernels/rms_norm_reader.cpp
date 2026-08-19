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
//    NOT legal: the second face read starts 32 B into the stick and Blackhole's
//    DRAM NoC alignment is 64 B.)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// ---------------------------------------------------------------------------
// gamma_row0 bake-off: the per-stage zones are compiled OUT in this copy.
// The bake-off metric is DEVICE KERNEL DURATION [ns], which comes from the
// always-on *-KERNEL firmware markers, not from these optional user zones.  The
// device profiler hashes each zone source location into 16 bits and THROWS on a
// collision; with several parallel experiment copies of these kernels in one
// build log the table gets dense enough to collide.  Dropping the optional zones
// also keeps their marker cost out of every arm's measurement.
// ---------------------------------------------------------------------------
#undef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name) ((void)0)

namespace {

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma_tiles = 1;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_rm_in = 8;
constexpr uint32_t cb_gamma_rm = 11;

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

// =====================================================================
// gamma_row0 EXPERIMENT KNOBS (this file is a perf_experiments copy)
// =====================================================================
// GAMMA_READ selects how much of each TILE-layout gamma page is fetched:
//   0 = FULL   (the op's CURRENT approach; honest baseline) one whole-page
//               noc_async_read_tile per gamma tile -> GAMMA_TILE_BYTES.
//   1 = SPAN    ONE transaction covering [0, row-0 end) of the page: face 0
//               entirely plus face 1's row 0.  Same transaction COUNT as the
//               baseline, 3.8x fewer bytes (bf16: 544 of 2048).
//   2 = FACES   TWO transactions, exactly the two row-0 runs (bf16: 32 B at
//               page offset 0 and 32 B at offset 512).  32x fewer bytes, 2x
//               the transactions.
// The downstream multiply is BroadcastDim::Row, which reads tile row 0 only, so
// SPAN and FACES leave tile rows 1..31 of the L1 page UNWRITTEN.  GAMMA_PREFILL
// exists to prove that is safe rather than assume it.
constexpr uint32_t GAMMA_READ = get_compile_time_arg_val(23);
// 0 = leave the untouched part of the gamma page as-is (stale L1);
// 1 = zero the whole gamma CB region once at kernel start;
// 2 = POISON it with bf16/fp32 NaN bit patterns once at kernel start.
//     Arm 2 is a CORRECTNESS PROBE: if the op still matches torch with rows
//     1..31 full of NaN, the Row-broadcast consumer provably never reads them.
constexpr uint32_t GAMMA_PREFILL = get_compile_time_arg_val(24);
// Option 2: in Regime B, fetch gamma from DRAM exactly ONCE per core into an L1
// cache of just the row-0 runs, then refill cb_gamma_tiles per W-chunk with
// LOCAL L1->L1 copies instead of re-reading DRAM once per row-block.
constexpr uint32_t GAMMA_CACHE = get_compile_time_arg_val(25);
constexpr uint32_t cb_gamma_cache = 12;

// --- Row-0 byte geometry of a gamma tile page -------------------------------
// A non-block tile is 4 faces of 16x16 stored contiguously; face 0 covers
// rows 0-15 / cols 0-15 at page offset 0 and face 1 covers rows 0-15 /
// cols 16-31 at offset 256*elem.  Tile row 0 is therefore TWO runs of
// 16*elem bytes, at offsets 0 and 256*elem.
//
// A bfloat8_b tile is 1088 B: 64 exponent bytes (one per face-row: 4 faces x
// 16 rows) followed by 4 faces of 256 mantissa bytes.  Row 0 of face f needs
// exponent byte 16*f and mantissa run [64 + 256*f, +16).  Fetching the whole
// 64 B exponent header is free (it is one aligned run), so run 0 is [0, 80)
// and run 1 is [320, 336).  Both starts are 64 B multiples.
constexpr bool GAMMA_IS_BLOCK = (GAMMA_ELEM_SIZE == 0);
constexpr uint32_t G_FACE_ROW = GAMMA_IS_BLOCK ? 16u : 16u * GAMMA_ELEM_SIZE;
constexpr uint32_t G_EXP_BYTES = GAMMA_IS_BLOCK ? 64u : 0u;
constexpr uint32_t G_FACE = GAMMA_IS_BLOCK ? 256u : 256u * GAMMA_ELEM_SIZE;
constexpr uint32_t G_RUN0_LEN = G_EXP_BYTES + G_FACE_ROW;
constexpr uint32_t G_RUN1_OFF = G_EXP_BYTES + G_FACE;
constexpr uint32_t G_RUN1_LEN = G_FACE_ROW;
constexpr uint32_t G_SPAN_LEN = G_RUN1_OFF + G_RUN1_LEN;
// Option 2's L1 cache layout.  The two runs are NOT packed back-to-back: the
// DRAM->cache load has to keep the 64 B residue match (below), so each run gets
// its own 64 B-aligned slot.  bf16: 128 B per tile instead of 2048.
constexpr uint32_t G_CACHE_RUN1_OFF = (G_RUN0_LEN + 63u) & ~63u;
constexpr uint32_t G_CACHE_STRIDE = G_CACHE_RUN1_OFF + ((G_RUN1_LEN + 63u) & ~63u);
// Pages of cb_gamma_tiles (host plan: Wt_core in Regime A, WT_SCALE_BLOCK in B).
constexpr uint32_t GAMMA_CB_PAGES = REGIME_A ? Wt_core : WT_SCALE_BLOCK;

// NoC LEGALITY of the partial reads (Blackhole).  The sanitizer rule is a
// RESIDUE match, not "size is a multiple of the alignment":
//     (l1_addr & (NOC_DRAM_READ_ALIGNMENT_BYTES-1)) == (dram_addr & mask)
// with NOC_DRAM_READ_ALIGNMENT_BYTES = 64.  Every DRAM gamma page starts
// 64 B-aligned (aligned_page_size = align(tile_bytes, 64) = 2048 / 4096 / 1088,
// all multiples of 64), the CB page stride is the tile size (also a multiple of
// 64), and every run offset above (0, 512/1024/320) is a multiple of 64.  So
// src and dst residues are both 0 for every run on every supported gamma dtype;
// only the LENGTH is sub-alignment, which the rule does not constrain.  This is
// why it is legal here and was NOT legal for the ROW_MAJOR gamma path, where
// the second face read starts 32 B into a stick (residue 32 vs 0).
static_assert(G_RUN1_OFF % 64 == 0, "gamma row-0 second run must start 64 B-aligned");
// The block-format row-0 geometry above is bfloat8_b-SPECIFIC (a 64 B exponent
// header of one byte per face-row, then 256 B mantissa faces).  bfloat4_b packs a
// 32 B header and 128 B faces, so the same constants would silently fetch the
// wrong bytes.  Fail the build instead of the numerics if a new block gamma dtype
// ever reaches here.
static_assert(
    !GAMMA_IS_BLOCK || GAMMA_TILE_BYTES == 1088,
    "gamma row-0 partial read: block-format geometry is bfloat8_b-only (1088 B tile)");
// PRECONDITION (not assertable here - TILE_DIM is declared below): the run
// geometry assumes the standard 32x32 tile of four 16x16 faces, which is what
// this op's host plan hard-codes everywhere (TILE_DIM = 32).

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

constexpr auto input_args = TensorAccessorArgs<26>();
[[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

FORCE_INLINE uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Zero `n` bytes of L1 starting at `addr`.  Only ever called on the padded tail
// of a row-major stick (< 32 elements), so the byte loop is bounded and cheap.
// Fill `n` bytes of L1 at `addr` with a repeating 32-bit pattern.  Used only by
// the GAMMA_PREFILL correctness probe (once per kernel, never on a hot path).
FORCE_INLINE void fill_l1(uint32_t addr, uint32_t n, uint32_t pattern) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < n / 4; ++i) {
        p[i] = pattern;
    }
}

// Local L1 -> L1 word copy (option 2's cache refill).  A NoC loopback transfer
// would also be legal here (L1 alignment is 16 B and every run is 16 B-aligned),
// but the runs are 16-80 B: a word copy beats paying a NoC transaction issue.
// Both addresses are 16 B-aligned and the lengths are multiples of 4.
FORCE_INLINE void copy_l1(uint32_t src, uint32_t dst, uint32_t n) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t i = 0; i < n / 4; ++i) {
        d[i] = s[i];
    }
}

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

    const auto in_acc = TensorAccessor(input_args, src_addr);

    // ---- gamma_row0 experiment scaffolding ---------------------------------
    // The first core of the work split is the injector for the GAMMA_READ==8
    // ceiling probe.
    const bool is_injector = (start_row_block == 0);
    uint32_t cache_addr = 0;
    if constexpr (HAS_GAMMA && !GAMMA_IS_ROW_MAJOR) {
        if constexpr (GAMMA_PREFILL) {
            // Stamp the WHOLE cb_gamma_tiles region once, before any gamma read.
            // GAMMA_READ=0 overwrites every byte of it, so this arm is a no-op on
            // the baseline; on the partial arms rows 1..31 keep the stamp for the
            // whole kernel, which is exactly the property under test.
            constexpr uint32_t pat = (GAMMA_PREFILL == 1) ? 0u
                                     : (GAMMA_ELEM_SIZE == 2)
                                         ? 0x7FC07FC0u  // two bf16 NaNs
                                         : ((GAMMA_ELEM_SIZE == 4) ? 0x7FC00000u   // fp32 NaN
                                                                   : 0xFFFFFFFFu);  // block-float garbage
            fill_l1(get_write_ptr(cb_gamma_tiles), GAMMA_CB_PAGES * GAMMA_TILE_BYTES, pat);
        }
        if constexpr (GAMMA_CACHE) {
            const auto g_acc = TensorAccessor(gamma_args, gamma_addr);
            cb_reserve_back(cb_gamma_cache, 1);
            cache_addr = get_write_ptr(cb_gamma_cache);
            {
                MaybeDeviceZoneScope("rd_gamma_cache");
                uint32_t a = cache_addr;
                for (uint32_t i = 0; i < Wt_core; ++i) {
                    noc_async_read(g_acc.get_noc_addr(i), a, G_RUN0_LEN);
                    noc_async_read(g_acc.get_noc_addr(i, G_RUN1_OFF), a + G_CACHE_RUN1_OFF, G_RUN1_LEN);
                    a += G_CACHE_STRIDE;
                }
                noc_async_read_barrier();
            }
            cb_push_back(cb_gamma_cache, 1);
        }
    }

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
                    const uint32_t byte_off = (w0 + o) * TILE_DIM * GAMMA_ELEM_SIZE;
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
                        if constexpr (GAMMA_CACHE) {
                            // Option 2: refill from the resident L1 row-0 cache
                            // (loaded from DRAM once per core) instead of DRAM.
                            const uint32_t c = cache_addr + (w0 + i) * G_CACHE_STRIDE;
                            copy_l1(c, addr, G_RUN0_LEN);
                            copy_l1(c + G_CACHE_RUN1_OFF, addr + G_RUN1_OFF, G_RUN1_LEN);
                        } else if constexpr (GAMMA_READ == 0) {
                            noc_async_read_tile(w0 + i, g_acc, addr);
                        } else if constexpr (GAMMA_READ == 1) {
                            noc_async_read(g_acc.get_noc_addr(w0 + i), addr, G_SPAN_LEN);
                        } else if constexpr (GAMMA_READ == 8) {
                            // CEILING PROBE for option 3 (mcast / ledger B12), not a
                            // candidate: ONE core does the SPAN read from DRAM and every
                            // other core issues nothing.  Output is wrong on the
                            // non-injector cores by construction, but the DURATION is a
                            // tighter upper bound on what a multicast could reach than
                            // the all-cores-skip ablation: it keeps the injector's DRAM
                            // read and every core's CB lifecycle, and omits only the
                            // mcast writes and the semaphore handshake.
                            if (is_injector) {
                                noc_async_read(g_acc.get_noc_addr(w0 + i), addr, G_SPAN_LEN);
                            }
                        } else if constexpr (GAMMA_READ == 4) {
                            // /perf-measure ABLATION (not a candidate): keep the
                            // CB lifecycle and the barrier, issue no transfer at
                            // all.  The floor it measures is what a PERFECT gamma
                            // ingest (e.g. one mcast injector) could reach, so it
                            // bounds option 3 before any of it is built.
                        } else if constexpr (GAMMA_READ == 5) {
                            // SPAN, but through the ONE-PACKET issue path.
                            // noc_async_read's default max_page_size is
                            // NOC_MAX_BURST_SIZE+1, which sends every call down
                            // ncrisc_noc_fast_read_any_len (a length loop);
                            // naming a compile-time max_page_size <= the burst
                            // size dispatches to noc_async_read_one_packet, a
                            // strictly shorter issue sequence.  Same bytes, same
                            // transaction count, cheaper RISC issue - which is
                            // what an ISSUE-BOUND shape actually needs.
                            noc_async_read<G_SPAN_LEN>(g_acc.get_noc_addr(w0 + i), addr, G_SPAN_LEN);
                        } else if constexpr (GAMMA_READ == 6) {
                            noc_async_read<G_RUN0_LEN>(g_acc.get_noc_addr(w0 + i), addr, G_RUN0_LEN);
                            noc_async_read<G_RUN1_LEN>(
                                g_acc.get_noc_addr(w0 + i, G_RUN1_OFF), addr + G_RUN1_OFF, G_RUN1_LEN);
                        } else if constexpr (GAMMA_READ == 7) {
                            // CONTROL: the baseline's whole page, but issued through
                            // the one-packet path.  Separates "fewer bytes" from
                            // "cheaper issue" in the attribution.
                            noc_async_read<GAMMA_TILE_BYTES>(g_acc.get_noc_addr(w0 + i), addr, GAMMA_TILE_BYTES);
                        } else if constexpr (GAMMA_READ == 3) {
                            // NEGATIVE CONTROL: run 0 only, so tile columns 16-31
                            // of every gamma tile are never fetched.  This arm MUST
                            // fail the torch gate - it is what proves the partial-read
                            // machinery is actually live and not silently falling back
                            // to a whole-page read.
                            noc_async_read(g_acc.get_noc_addr(w0 + i), addr, G_RUN0_LEN);
                        } else {
                            noc_async_read(g_acc.get_noc_addr(w0 + i), addr, G_RUN0_LEN);
                            noc_async_read(
                                g_acc.get_noc_addr(w0 + i, G_RUN1_OFF), addr + G_RUN1_OFF, G_RUN1_LEN);
                        }
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
                const uint32_t row_base = umin(rt0 + r, LAST_RT) * Wt_core + w0;
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
        const uint32_t byte_off = w0 * TILE_DIM * ELEM_SIZE;
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
