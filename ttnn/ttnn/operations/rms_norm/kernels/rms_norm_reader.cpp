// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for rms_norm.
//
// Responsibilities:
//   1) Emit the reduce scaler tile(s) to cb_scaler exactly once.
//      - Without partial W: 1 scaler tile (1/W in matmul col-0 layout).
//      - With partial W:    2 tiles (full + partial; partial zeroes padded columns).
//   2) Stream the input data per row-chunk:
//      - INPUT_IS_RM: 32 rows per chunk via read_sticks_for_tilize<TILE>.
//        (Compute kernel tilizes them into cb_input_tiles.)
//      - !INPUT_IS_RM (TILE input): Wt tiles per chunk via noc_async_read_tile,
//        pushed directly into cb_input_tiles.
//   3) Stream gamma per row-chunk (gamma path only):
//      - Push 1 row of gamma into cb_gamma_rm per chunk.
//      - Compute tilizes (asymmetric, 1 row → Wt tiles).
//
// Reader runs on NCRISC (RISCV_1). It pushes data and never reads from a CB.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(0);
    constexpr uint32_t HAS_PARTIAL_W = get_compile_time_arg_val(1);
    constexpr uint32_t partial_w = get_compile_time_arg_val(2);
    constexpr uint32_t INPUT_IS_RM = get_compile_time_arg_val(3);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t input_row_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t gamma_row_bytes = get_compile_time_arg_val(8);

    // TensorAccessorArgs MUST be declared unconditionally (CT slot mapping).
    constexpr auto input_args = TensorAccessorArgs<9>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // ---- Runtime args ----
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_unit = get_arg_val<uint32_t>(2);   // start row (RM) or start tile (TILE)
    const uint32_t total_units = get_arg_val<uint32_t>(3);  // total rows (RM) or total tiles (TILE)

    // ---- CB ids (must match program descriptor) ----
    constexpr uint32_t cb_input_raw_rm = 0;
    constexpr uint32_t cb_input_tiles = 1;
    constexpr uint32_t cb_gamma_rm = 2;
    constexpr uint32_t cb_scaler = 4;

    // ---- Step 1: emit scaler tile(s) once ----
    // Use pool-type-aware overloads (system rule): REDUCE_ROW + SUM uses matmul-path col-0 fill.
    const float scaler_f = __builtin_bit_cast(float, scaler_bits);
    if constexpr (HAS_PARTIAL_W) {
        dataflow_kernel_lib::prepare_partial_reduce_scalers<
            cb_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            partial_w>(scaler_f);
    } else {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scaler_f);
    }

    // ---- Step 2: per-chunk streaming loop ----
    // Compute kernel consumes gamma + input per chunk. To avoid producer/consumer
    // deadlock, the reader pushes them in the same order, per chunk:
    //   for each chunk: push gamma (1 stick) → push input (Wt tiles or 32 sticks)
    // An aggregated read_sticks_for_tilize over all chunks would fill cb_input_raw_rm
    // before any gamma is pushed, and compute would block on cb_gamma_rm while the
    // reader blocks on a full cb_input_raw_rm.

    constexpr uint32_t TILE_DIM = 32;

    // Input accessor — RM uses stick-indexed (no tile_bytes), TILE uses tile-indexed.
    const auto input_accessor = [&] {
        if constexpr (INPUT_IS_RM) {
            return TensorAccessor(input_args, input_addr);
        } else {
            return TensorAccessor(input_args, input_addr, get_tile_size(cb_input_tiles));
        }
    }();

    [[maybe_unused]] const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr);
    [[maybe_unused]] const uint64_t gamma_noc_addr = HAS_GAMMA ? gamma_accessor.get_noc_addr(0) : 0;

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        // Push gamma BEFORE input — matches compute's consumption order (compute tilizes
        // gamma first per chunk, then input). cb_gamma_rm sized 1 → reader blocks here
        // until compute pops the previous chunk's gamma, which is the natural backpressure.
        if constexpr (HAS_GAMMA) {
            cb_reserve_back(cb_gamma_rm, 1);
            const uint32_t l1_write_addr = get_write_ptr(cb_gamma_rm);
            noc_async_read(gamma_noc_addr, l1_write_addr, gamma_row_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_gamma_rm, 1);
        }

        // Push input for this chunk.
        if constexpr (INPUT_IS_RM) {
            // Compute valid rows for this chunk (handles partial-H on the last chunk).
            const uint32_t chunk_first_row = start_unit + chunk * TILE_DIM;
            const uint32_t global_end = start_unit + total_units;
            const uint32_t rows_remaining = (chunk_first_row < global_end) ? (global_end - chunk_first_row) : 0;
            const uint32_t rows_this_chunk = (rows_remaining < TILE_DIM) ? rows_remaining : TILE_DIM;

            // Single block per chunk. Helper pushes Wt tile-sized pages even on partial-H
            // (rows beyond rows_this_chunk are stale; compute produces garbage there;
            // writer skips them via write_sticks_after_untilize).
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_raw_rm, dataflow_kernel_lib::TilizeGranularity::TILE>(
                input_accessor, rows_this_chunk, input_row_bytes, chunk_first_row);
        } else {
            // TILE input: Wt tiles per chunk.
            const uint32_t chunk_first_tile = start_unit + chunk * Wt;
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_reserve_back(cb_input_tiles, 1);
                const uint32_t l1_write_addr = get_write_ptr(cb_input_tiles);
                noc_async_read_tile(chunk_first_tile + wt, input_accessor, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_input_tiles, 1);
            }
        }
    }
}
