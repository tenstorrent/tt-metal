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

// --- lab-only zone gate ------------------------------------------------------
// EVERY per-stage zone call site is spelled `RSR_ZONE(x)`.  With -DRMSN_NO_ZONES
// the RAII zone becomes a no-op, so a chunk/depth sweep measured under --profile
// is not paying a marker cost that differs between arms (Regime A/C run a
// DIFFERENT NUMBER of zone executions than Regime B, which would otherwise leak
// straight into the ns delta being attributed to the idea).  Zones ON is still
// available (`no_zones=0`) for the per-stage breakdown.
#ifdef RMSN_NO_ZONES
#define RSR_ZONE(name) ((void)0)
#else
#define RSR_ZONE(name) MaybeDeviceZoneScope(name)
#endif

namespace {

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma_tiles = 1;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_rm_in = 8;
constexpr uint32_t cb_gamma_rm = 11;

// --- shared geometry compile-time args (identical prefix in all 3 kernels) ---
constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(0);
// REGIME (CT arg 1) — three plans, not two:
//   0 = B  STREAMING-MASKED   two DRAM reads of x (reduce pass + scale pass).
//   1 = A  RESIDENT-FUSED     one DRAM read; the WHOLE per-core width of x AND
//                             gamma AND cb_normed AND the output CB are resident.
//   2 = C  RESIDENT-X         one DRAM read; ONLY x is resident, and the scale
//                             pass walks W in WT_SCALE_BLOCK chunks (gamma /
//                             normed / output CBs are sized per chunk).
constexpr uint32_t REGIME = get_compile_time_arg_val(1);
constexpr bool REGIME_A = (REGIME == 1);
constexpr bool REGIME_C = (REGIME == 2);
// x is read from DRAM exactly ONCE per row-block and stays in cb_input_tiles for
// the scale pass.  This is the property the experiment is about.
constexpr bool RESIDENT_X = REGIME_A || REGIME_C;
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
// Regime C decomposition arm (CT 23): 1 = the FUSED sum_of_squares reduce (Regime
// A's datapath), 0 = Regime B's STREAMING square->accumulating-reduce datapath run
// over the RESIDENT x.  Splits "one DRAM read" from "fused reduce" so the two
// mechanisms can be attributed separately.  Reader-side it only matters for the
// scaler tile the reduce consumes.
constexpr uint32_t C_FUSED_REDUCE = get_compile_time_arg_val(23);
// Regime C gamma residency (CT 24): 1 = gamma stays RESIDENT at full per-core
// width and is read ONCE per core (Regime A's protocol, which also keeps A's
// "cb_gamma_tiles is never popped" invariant); 0 = gamma is chunked and re-pushed
// per W-chunk per row-block (Regime B's protocol).  With more than one row-block
// per core, chunked gamma is re-read once per row-block, which on a prefill shape
// is the SAME transaction count as x itself.
constexpr uint32_t C_RESIDENT_GAMMA = get_compile_time_arg_val(24);
constexpr bool GAMMA_RESIDENT = REGIME_A || (REGIME_C && C_RESIDENT_GAMMA);

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

constexpr auto input_args = TensorAccessorArgs<25>();
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
        RSR_ZONE("rd_scaler");
        if constexpr (!RESIDENT_X && W_PARTIAL > 0) {
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
                        RSR_ZONE("rd_gamma_reserve");
                        cb_reserve_back(cb_gamma_rm, GAMMA_INGEST_BLOCK);
                    }
                    const uint32_t addr = get_write_ptr(cb_gamma_rm);
                    if (byte_off < GAMMA_ROW_BYTES) {
                        {
                            RSR_ZONE("rd_gamma_issue");
                            noc_async_read(
                                g_acc.get_noc_addr(0, byte_off), addr, umin(group_bytes, GAMMA_ROW_BYTES - byte_off));
                        }
                        {
                            RSR_ZONE("rd_gamma_barrier");
                            noc_async_read_barrier();
                        }
                    }
                    cb_push_back(cb_gamma_rm, GAMMA_INGEST_BLOCK);
                }
            } else {
                {
                    RSR_ZONE("rd_gamma_reserve");
                    cb_reserve_back(cb_gamma_tiles, n);
                }
                uint32_t addr = get_write_ptr(cb_gamma_tiles);
                {
                    RSR_ZONE("rd_gamma_issue");
                    for (uint32_t i = 0; i < n; ++i) {
                        noc_async_read_tile(w0 + i, g_acc, addr);
                        addr += GAMMA_TILE_BYTES;
                    }
                }
                {
                    RSR_ZONE("rd_gamma_barrier");
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
            RSR_ZONE("rd_in_reserve");
            cb_reserve_back(cb_input_tiles, n);
        }
        uint32_t addr = get_write_ptr(cb_input_tiles);
        {
            RSR_ZONE("rd_in_issue");
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
            RSR_ZONE("rd_in_barrier");
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
            RSR_ZONE("rd_in_reserve");
            cb_reserve_back(cb_rm_in, nw);
        }
        const uint32_t base = get_write_ptr(cb_rm_in);
        uint32_t dst = base;
        {
            RSR_ZONE("rd_in_issue");
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
            RSR_ZONE("rd_in_barrier");
            noc_async_read_barrier();
        }
        // Zero the pad tail of every valid stick so tilize never promotes
        // uninitialised L1 into the reduction.  H-padding rows need no fill:
        // the reduction is per-row and the writer never emits a pad row.
        if (chunk_bytes < padded) {
            RSR_ZONE("rd_zero_pad");
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
    // Regime A keeps its boot-time gamma fill (untouched control).  Regime C's
    // resident gamma is deliberately filled LATER - see the row-block loop.
    if constexpr (GAMMA_RESIDENT && !REGIME_C) {
        fill_gamma(0, Wt_core);
    }

    for (uint32_t b = 0; b < num_row_blocks_here; ++b) {
        const uint32_t rt0 = (start_row_block + b) * BLOCK_HT;

        if constexpr (REGIME_A) {
            read_input_chunk(rt0, 0, Wt_core);
        } else if constexpr (REGIME_C) {
            // THE IDEA: ONE full-width read of x per row-block (as in Regime A),
            // then the scale pass consumes it from L1 in chunks.  The only thing
            // the reader still streams per chunk is this chunk's gamma slice -
            // exactly Regime B's gamma protocol, which is why Regime C is a
            // MERGE of the two regimes rather than a third code path.
            read_input_chunk(rt0, 0, Wt_core);
            if constexpr (GAMMA_RESIDENT) {
                // ONE gamma read for the whole core, issued AFTER the first
                // row-block's x.  Order matters and is measurable: gamma is only
                // needed by the SCALE pass, so issuing it after x lets it overlap
                // the sum-of-squares compute.  Filling it first (the Regime A
                // order) serialises Wt_core gamma transactions ahead of the x
                // read that compute is actually waiting on - measured +7.2 us on
                // the focus shape (37,397 vs 30,230 ns).
                if (b == 0) {
                    fill_gamma(0, Wt_core);
                }
            } else {
                for (uint32_t c = 0; c < NUM_SCALE_CHUNKS; ++c) {
                    fill_gamma(c * WT_SCALE_BLOCK, WT_SCALE_BLOCK);
                }
            }
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
