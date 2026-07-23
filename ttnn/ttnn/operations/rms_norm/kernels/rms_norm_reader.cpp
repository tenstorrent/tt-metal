// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for rms_norm.
//
// Row-parallel: this core owns tile-rows [row_start, row_start+num_rows).
// Each tile-row is streamed TWICE (2-pass reduce): pass 0 feeds x^2->reduce,
// pass 1 feeds x*rstd and (optional) gamma.
//
// Two layout regimes (selected by the compile-time IS_RM flag):
//   TILE : read whole tiles directly into cb_x_in (tile_id = r*Wt + b*BS + wt).
//   RM   : read row-major sticks into cb_x_sticks via the tilize dataflow
//          helper (compute tilizes them). The helper handles non-tile-aligned
//          W (row_bytes), non-tile-aligned H (partial last block), the per-core
//          start-row offset, and W-block chunking (byte offset).
//
// gamma (1,1,1,W) is read per-block in pass 1, in the order compute consumes it.
// Two gamma regimes (selected by the compile-time GAMMA_IS_RM flag):
//   RM gamma   : read row-major sticks into cb_gamma_sticks (compute tilizes).
//   TILE gamma : read whole tiles straight into cb_gamma (compute skips tilize),
//                same batched-read path as the TILE input (Refinement 2 knob-turn).
//
// Raw TensorAccessor + noc_async_read_tile is used for the TILE path (no
// kernel-lib helper covers the custom two-pass, per-core tile_id order).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_x_sticks = 0;
constexpr uint32_t cb_x_in = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_gamma = 3;
constexpr uint32_t cb_gamma_sticks = 4;
constexpr uint32_t TILE_DIM = 32;
}  // namespace

void kernel_main() {
    constexpr uint32_t Ht_img = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(3);
    constexpr uint32_t origin_H = get_compile_time_arg_val(4);
    constexpr uint32_t origin_W = get_compile_time_arg_val(5);
    constexpr uint32_t inv_N_bits = get_compile_time_arg_val(6);
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t partial_w = get_compile_time_arg_val(8);
    constexpr bool IS_RM = get_compile_time_arg_val(9) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t in_elem = get_compile_time_arg_val(11);
    constexpr uint32_t gamma_elem = get_compile_time_arg_val(12);
    constexpr uint32_t in_page = get_compile_time_arg_val(13);
    constexpr uint32_t gamma_page = get_compile_time_arg_val(14);
    constexpr bool GAMMA_IS_RM = get_compile_time_arg_val(15) != 0;
    // Resident single-read fast-path (op_design.md §1 lamp 1, Refinement 3): TILE
    // input (+ TILE / no gamma) that fits L1. x is read ONCE per tile-row (compute
    // does both passes over L1) and gamma is read ONCE per core (held resident).
    constexpr bool USE_RESIDENT = get_compile_time_arg_val(16) != 0;
    // HEIGHT_SHARDED local per-core reduction (op_design.md §1 lamp 3, Refinement 5):
    // this core's full-W row-shard is already resident in its L1, so cb_x_in is a
    // zero-copy CB backed on the sharded buffer (compute self-arms it). The reader
    // does NOT read x at all — only the scaler prep and the (interleaved DRAM) gamma
    // read below run. A knob-turn on the USE_RESIDENT path: same indexed two-pass
    // compute, x placement pinned by the shard instead of read from DRAM.
    constexpr bool X_RESIDENT = get_compile_time_arg_val(17) != 0;

    constexpr auto in_args = TensorAccessorArgs<18>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);

    // --- scaler prep (once; wait-not-pop across all rows/passes) ---
    // scaler = 1/origin_W so SUM-reduce produces mean(x^2) directly.
    // Deviation (forced, advisory): prepare_partial_reduce_scalers is stale in
    // this kernel_lib checkout (it forwards a 4th `compute_uses_reduce_tile`
    // template arg to prepare_reduce_scaler, which takes 3). We emit the same
    // full(tile0)+partial(tile1) pair it would produce, via two direct
    // prepare_reduce_scaler calls (reduce_helpers_dataflow.inl:333-352). Pairs
    // with ReducePartialScaler::last_tile_at(1) on the compute side.
    const float scaler_f = __builtin_bit_cast(float, inv_N_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scaler_f);  // tile 0: full scaler
    if constexpr (HAS_PARTIAL_W) {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scaler_f, partial_w);  // tile 1: partial scaler (zeros padded lanes)
    }

    const auto in_accessor = TensorAccessor(in_args, src_addr, in_page);
    const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_page);

    // RM + non-tile-aligned W: the tilize dataflow helper leaves the padded
    // columns of the last W-tile as stale L1. Those columns are multiplied by
    // the partial scaler's zero in the reduce (inf*0 = nan), so zero the whole
    // cb_x_sticks backing once up-front — reads only overwrite the valid
    // [0:row_bytes] region, so the pad stays 0 for the whole kernel.
    if constexpr (IS_RM && HAS_PARTIAL_W) {
        const uint32_t base = get_write_ptr(cb_x_sticks);
        const uint32_t nbytes = get_local_cb_interface(cb_x_sticks).fifo_num_pages * get_tile_size(cb_x_sticks);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
        for (uint32_t k = 0; k < (nbytes >> 2); ++k) {
            p[k] = 0;
        }
    }

    // ---- Resident single-read fast-path (TILE input, TILE/no gamma) ----
    // gamma read ONCE (held resident across all rows); x read ONCE per tile-row
    // (compute does BOTH passes over L1). One coalesced barrier per read.
    if constexpr (USE_RESIDENT) {
        if constexpr (HAS_GAMMA && !X_RESIDENT) {
            // Interleaved resident: gamma (1,1,1,W) -> Wt tiles in one tile-row; read all
            // Wt ONCE, held resident across all rows.
            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
            cb_reserve_back(cb_gamma, Wt);
            uint32_t gl1 = get_write_ptr(cb_gamma);
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                noc_async_read_tile(wt, gamma_accessor, gl1);
                gl1 += gamma_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma, Wt);
        }
        if constexpr (!X_RESIDENT) {
            // Interleaved resident: x read ONCE per tile-row (compute does both passes over L1).
            const uint32_t tile_bytes = get_tile_size(cb_x_in);
            for (uint32_t i = 0; i < num_rows; ++i) {
                const uint32_t row_tile_base = (row_start + i) * Wt;
                cb_reserve_back(cb_x_in, Wt);
                uint32_t l1 = get_write_ptr(cb_x_in);
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    noc_async_read_tile(row_tile_base + wt, in_accessor, l1);
                    l1 += tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_x_in, Wt);
            }
        } else if constexpr (HAS_GAMMA) {
            // HEIGHT_SHARDED (Refinement 5): x is the resident zero-copy shard (compute
            // self-arms cb_x_in), so skip the x read. gamma is STREAMED per block per row
            // in compute's pass-2 consume order (small cb_gamma fits any W; gamma is the
            // same (1,1,1,W) for every row so tile b = b*BLOCK_SIZE + wt, re-read per row).
            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
            for (uint32_t i = 0; i < num_rows; ++i) {
                for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                    cb_reserve_back(cb_gamma, BLOCK_SIZE);
                    uint32_t gl1 = get_write_ptr(cb_gamma);
                    for (uint32_t wt = 0; wt < BLOCK_SIZE; ++wt) {
                        noc_async_read_tile(b * BLOCK_SIZE + wt, gamma_accessor, gl1);
                        gl1 += gamma_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_gamma, BLOCK_SIZE);
                }
            }
        }
        return;
    }

    for (uint32_t i = 0; i < num_rows; ++i) {
        const uint32_t r = row_start + i;
        const uint32_t image = r / Ht_img;
        const uint32_t ht_in_img = r % Ht_img;
        const uint32_t base_stick = image * origin_H + ht_in_img * TILE_DIM;  // RM
        uint32_t valid_rows = origin_H - ht_in_img * TILE_DIM;                // RM
        if (valid_rows > TILE_DIM) {
            valid_rows = TILE_DIM;
        }
        const uint32_t row_tile_base = r * Wt;  // TILE

        for (uint32_t pass = 0; pass < 2; ++pass) {
            for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                if constexpr (IS_RM) {
                    const uint32_t col0 = b * BLOCK_SIZE * TILE_DIM;
                    uint32_t cols = origin_W - col0;
                    if (cols > BLOCK_SIZE * TILE_DIM) {
                        cols = BLOCK_SIZE * TILE_DIM;
                    }
                    dataflow_kernel_lib::read_sticks_for_tilize<cb_x_sticks>(
                        in_accessor, valid_rows, cols * in_elem, base_stick, col0 * in_elem);
                } else {
                    // Coalesce the whole block behind ONE barrier (keep BLOCK_SIZE
                    // reads in flight -> DRAM-bandwidth, not per-tile latency).
                    const uint32_t tile_bytes = get_tile_size(cb_x_in);
                    cb_reserve_back(cb_x_in, BLOCK_SIZE);
                    uint32_t l1 = get_write_ptr(cb_x_in);
                    for (uint32_t wt = 0; wt < BLOCK_SIZE; ++wt) {
                        noc_async_read_tile(row_tile_base + b * BLOCK_SIZE + wt, in_accessor, l1);
                        l1 += tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_x_in, BLOCK_SIZE);
                }

                // gamma: pass 2 only (pass==1), per block, in compute's consume order.
                if constexpr (HAS_GAMMA) {
                    if (pass == 1) {
                        if constexpr (GAMMA_IS_RM) {
                            const uint32_t col0 = b * BLOCK_SIZE * TILE_DIM;
                            uint32_t cols = origin_W - col0;
                            if (cols > BLOCK_SIZE * TILE_DIM) {
                                cols = BLOCK_SIZE * TILE_DIM;
                            }
                            // gamma is (1,1,1,W): one stick; only row 0 is read by the
                            // Row-broadcast multiply, so total_num_rows = 1.
                            dataflow_kernel_lib::read_sticks_for_tilize<cb_gamma_sticks>(
                                gamma_accessor, 1, cols * gamma_elem, 0, col0 * gamma_elem);
                        } else {
                            // TILE gamma: already tiled, (1,1,1,W) -> Wt tiles in one
                            // tile-row, so tile_id = b*BLOCK_SIZE + wt. Coalesce the
                            // whole block behind ONE barrier (writer/reader batched-read
                            // twin -> DRAM bandwidth, not per-tile latency).
                            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
                            cb_reserve_back(cb_gamma, BLOCK_SIZE);
                            uint32_t l1 = get_write_ptr(cb_gamma);
                            for (uint32_t wt = 0; wt < BLOCK_SIZE; ++wt) {
                                noc_async_read_tile(b * BLOCK_SIZE + wt, gamma_accessor, l1);
                                l1 += gamma_tile_bytes;
                            }
                            noc_async_read_barrier();
                            cb_push_back(cb_gamma, BLOCK_SIZE);
                        }
                    }
                }
            }
        }
    }
}
