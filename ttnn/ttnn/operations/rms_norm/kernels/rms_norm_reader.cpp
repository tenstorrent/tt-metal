// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm reader (NCRISC / NoC0).
//
// Feeds, per row-block owned by this core:
//   TILE input      -> cb_input_tiles   (BLOCK_HT * chunk_wt tiles, one barrier per block)
//   ROW_MAJOR input -> cb_rm_in         (chunk_wt tile-pages per tile-row of 32 sticks)
//   gamma           -> cb_gamma_tiles   (resident in Regime A, per-chunk in Regime B)
//   reduce scaler   -> cb_reduce_scaler (Regime B only, once per core)
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
//    become NaN through the masked scaler (risk R3).  The helper exposes no
//    zero-fill hook and no per-chunk byte window combined with it, so the loop
//    is written out here.  It is otherwise structurally identical to the
//    helper's TILE-granularity path: one cb_reserve_back + N reads + ONE
//    noc_async_read_barrier + one cb_push_back per tile-row (lever B7).
//
// 2. ROW_MAJOR gamma is NOT ingested through a `cb_gamma_rm` staging CB plus a
//    compute-side tilize().  That path needs 32 stick-rows of the full padded
//    width resident (Wt * tile_bytes) purely to fill tile rows 1..31 that the
//    downstream `BroadcastDim::Row` multiply never reads - 1 MB of L1 for dead
//    rows at W = 16384, which makes the acceptance test's wide cases
//    unschedulable.  Instead the reader places the gamma stick directly into
//    tile row 0 of each gamma tile, which is exactly the region
//    BroadcastDim::Row consumes.  Tile row 0 lives at byte 0 (face 0, columns
//    0..15) and byte 256*elem (face 1, columns 16..31) of a 32x32 tile.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma_tiles = 1;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_rm_in = 8;

// --- shared geometry compile-time args (identical prefix in all 3 kernels) ---
constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(0);
constexpr uint32_t REGIME_A = get_compile_time_arg_val(1);
constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(2);
constexpr uint32_t GAMMA_IS_ROW_MAJOR = get_compile_time_arg_val(3);
constexpr uint32_t Wt_core = get_compile_time_arg_val(4);
constexpr uint32_t W_PARTIAL = get_compile_time_arg_val(5);
constexpr uint32_t BLOCK_HT = get_compile_time_arg_val(6);
constexpr uint32_t WT_REDUCE_BLOCK = get_compile_time_arg_val(7);
constexpr uint32_t WT_REDUCE_TAIL = get_compile_time_arg_val(8);
constexpr uint32_t WT_SCALE_BLOCK = get_compile_time_arg_val(9);
constexpr uint32_t WT_SCALE_TAIL = get_compile_time_arg_val(10);
constexpr uint32_t Rt = get_compile_time_arg_val(11);
constexpr uint32_t NUM_ROWS = get_compile_time_arg_val(12);
constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(13);
constexpr uint32_t ELEM_SIZE = get_compile_time_arg_val(14);
constexpr uint32_t GAMMA_ELEM_SIZE = get_compile_time_arg_val(15);
constexpr uint32_t GAMMA_ROW_BYTES = get_compile_time_arg_val(16);
constexpr uint32_t DEST_BLOCK_CT = get_compile_time_arg_val(17);
constexpr uint32_t GAMMA_TILE_BYTES = get_compile_time_arg_val(18);
constexpr uint32_t IN_TILE_BYTES = get_compile_time_arg_val(19);

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t NUM_REDUCE_CHUNKS = (Wt_core + WT_REDUCE_BLOCK - 1) / WT_REDUCE_BLOCK;
constexpr uint32_t NUM_SCALE_CHUNKS = (Wt_core + WT_SCALE_BLOCK - 1) / WT_SCALE_BLOCK;

constexpr auto input_args = TensorAccessorArgs<20>();
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
    {
        if constexpr (!REGIME_A && W_PARTIAL > 0) {
            dataflow_kernel_lib::prepare_partial_reduce_scalers<
                cb_reduce_scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                W_PARTIAL>(1.0f);
        } else {
            dataflow_kernel_lib::
                prepare_reduce_scaler<cb_reduce_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
        }
    }

    // ---- gamma ingest -------------------------------------------------------
    // Places gamma tiles [w0, w0 + n) into cb_gamma_tiles.
    auto fill_gamma = [&](uint32_t w0, uint32_t n) {
        if constexpr (HAS_GAMMA) {
            const auto g_acc = TensorAccessor(gamma_args, gamma_addr);
            cb_reserve_back(cb_gamma_tiles, n);
            uint32_t addr = get_write_ptr(cb_gamma_tiles);
            if constexpr (GAMMA_IS_ROW_MAJOR) {
                constexpr uint32_t half_bytes = 16 * GAMMA_ELEM_SIZE;
                constexpr uint32_t face1_off = 256 * GAMMA_ELEM_SIZE;
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t off0 = (w0 + i) * TILE_DIM * GAMMA_ELEM_SIZE;
                    if (off0 < GAMMA_ROW_BYTES) {
                        noc_async_read(g_acc.get_noc_addr(0, off0), addr, umin(half_bytes, GAMMA_ROW_BYTES - off0));
                    }
                    const uint32_t off1 = off0 + half_bytes;
                    if (off1 < GAMMA_ROW_BYTES) {
                        noc_async_read(
                            g_acc.get_noc_addr(0, off1), addr + face1_off, umin(half_bytes, GAMMA_ROW_BYTES - off1));
                    }
                    addr += GAMMA_TILE_BYTES;
                }
            } else {
                for (uint32_t i = 0; i < n; ++i) {
                    noc_async_read_tile(w0 + i, g_acc, addr);
                    addr += GAMMA_TILE_BYTES;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma_tiles, n);
        }
    };

    // ---- TILE input: one row-block (or W-chunk of it) per call --------------
    auto read_tiles = [&](uint32_t rt0, uint32_t ht, uint32_t w0, uint32_t nw) {
        const uint32_t n = ht * nw;
        cb_reserve_back(cb_input_tiles, n);
        uint32_t addr = get_write_ptr(cb_input_tiles);
        for (uint32_t r = 0; r < ht; ++r) {
            const uint32_t row_base = (rt0 + r) * Wt_core + w0;
            for (uint32_t w = 0; w < nw; ++w) {
                noc_async_read_tile(row_base + w, in_acc, addr);
                addr += IN_TILE_BYTES;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, n);
    };

    // ---- ROW_MAJOR input: one tile-row (32 sticks) of a W-chunk per call ----
    auto read_sticks = [&](uint32_t rt, uint32_t w0, uint32_t nw) {
        const uint32_t row0 = rt * TILE_DIM;
        const uint32_t nrows = umin(TILE_DIM, NUM_ROWS - row0);
        const uint32_t byte_off = w0 * TILE_DIM * ELEM_SIZE;
        const uint32_t padded = nw * TILE_DIM * ELEM_SIZE;
        const uint32_t chunk_bytes = umin(padded, ROW_BYTES - byte_off);

        cb_reserve_back(cb_rm_in, nw);
        const uint32_t base = get_write_ptr(cb_rm_in);
        uint32_t dst = base;
        for (uint32_t r = 0; r < nrows; ++r) {
            noc_async_read(in_acc.get_noc_addr(row0 + r, byte_off), dst, chunk_bytes);
            dst += padded;
        }
        noc_async_read_barrier();
        // Zero the pad tail of every valid stick so tilize never promotes
        // uninitialised L1 into the reduction.  H-padding rows need no fill:
        // the reduction is per-row and the writer never emits a pad row.
        if (chunk_bytes < padded) {
            uint32_t tail = base + chunk_bytes;
            for (uint32_t r = 0; r < nrows; ++r) {
                zero_l1(tail, padded - chunk_bytes);
                tail += padded;
            }
        }
        cb_push_back(cb_rm_in, nw);
    };

    auto read_input_chunk = [&](uint32_t rt0, uint32_t ht, uint32_t w0, uint32_t nw) {
        if constexpr (IS_ROW_MAJOR) {
            for (uint32_t r = 0; r < ht; ++r) {
                read_sticks(rt0 + r, w0, nw);
            }
        } else {
            read_tiles(rt0, ht, w0, nw);
        }
    };

    // Regime A holds the whole per-core width of gamma resident for the whole
    // kernel: filled exactly once, never popped.  That is what makes the gamma
    // read cost 1x per core instead of 1x per row-block.
    if constexpr (REGIME_A) {
        fill_gamma(0, Wt_core);
    }

    for (uint32_t b = 0; b < num_row_blocks_here; ++b) {
        const uint32_t rt0 = (start_row_block + b) * BLOCK_HT;
        const uint32_t ht = umin(BLOCK_HT, Rt - rt0);

        if constexpr (REGIME_A) {
            read_input_chunk(rt0, ht, 0, Wt_core);
        } else {
            // pass A - reduction
            for (uint32_t c = 0; c < NUM_REDUCE_CHUNKS; ++c) {
                const uint32_t nw = (c + 1 == NUM_REDUCE_CHUNKS) ? WT_REDUCE_TAIL : WT_REDUCE_BLOCK;
                read_input_chunk(rt0, ht, c * WT_REDUCE_BLOCK, nw);
            }
            // pass B - scale (re-read of x, plus this chunk's gamma slice)
            for (uint32_t c = 0; c < NUM_SCALE_CHUNKS; ++c) {
                const uint32_t nw = (c + 1 == NUM_SCALE_CHUNKS) ? WT_SCALE_TAIL : WT_SCALE_BLOCK;
                fill_gamma(c * WT_SCALE_BLOCK, nw);
                read_input_chunk(rt0, ht, c * WT_SCALE_BLOCK, nw);
            }
        }
    }
}
