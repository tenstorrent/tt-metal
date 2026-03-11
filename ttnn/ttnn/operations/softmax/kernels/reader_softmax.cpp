// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax - Reader Kernel
// 3-pass reader for softmax: reads each row/col of tiles 3 times from DRAM.
// Pass 1: for max computation
// Pass 2: for sub+exp+sum accumulation
// Pass 3: for final multiply
// Also generates constant scaler tiles c_1 (reduce scaler) and c_2 (mm scaler).

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp"

constexpr uint32_t cb_input = tt::CBIndex::c_0;
constexpr uint32_t cb_scaler = tt::CBIndex::c_1;
constexpr uint32_t cb_mm_scaler = tt::CBIndex::c_2;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t num_rows_or_cols = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(3);
    constexpr auto tensor_accessor_args = TensorAccessorArgs<4>();

    // Runtime args
    const uint32_t input_addr = get_arg_val<uint32_t>(0);

    // Setup tensor accessor
    const uint32_t page_size = get_tile_size(cb_input);
    const auto input_accessor = TensorAccessor(tensor_accessor_args, input_addr, page_size);

    // Generate constant scaler tiles at startup
    // c_1: reduce scaler (all 1.0 in row 0 of each face)
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f);

    // c_2: matmul row-reduce scaler (column vector of 1.0 in left faces)
    constexpr uint32_t packed_bf16_1_0 = 0x3F803F80u;
    generate_mm_scaler(cb_mm_scaler, packed_bf16_1_0);

    // 3-pass read: for each row, read the same Wt tiles 3 times
    // dim=-1: row = NC*Ht rows, each Wt tiles wide, tile_id = row * Wt + wt
    // dim=-2: col = NC*Wt columns, each Ht tiles tall, tile_id = ht * Wt + col
    // The compute kernel consumes 3 * inner_dim tiles per row/col iteration.

#ifdef DIM_W
    // dim=-1: iterate over rows, each row is Wt tiles
    for (uint32_t row = 0; row < num_rows_or_cols; ++row) {
        uint32_t row_start_tile = row * Wt;
        // 3 passes over the same row
        for (uint32_t pass = 0; pass < 3; ++pass) {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                uint32_t tile_id = row_start_tile + wt;
                cb_reserve_back(cb_input, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_input);
                uint64_t noc_addr = input_accessor.get_noc_addr(tile_id);
                noc_async_read(noc_addr, l1_write_addr, page_size);
                noc_async_read_barrier();
                cb_push_back(cb_input, 1);
            }
        }
    }
#endif

#ifdef DIM_H
    // dim=-2: iterate over columns, each column is Ht tiles
    // Column-major access: for column wt, tiles at ht*Wt + wt for ht in 0..Ht-1
    for (uint32_t col = 0; col < num_rows_or_cols; ++col) {
        // col indexes into NC*Wt flat space
        // batch = col / Wt, wt_in_batch = col % Wt
        uint32_t batch = col / Wt;
        uint32_t wt_in_batch = col % Wt;
        uint32_t batch_start_tile = batch * Ht * Wt;
        // 3 passes over the same column
        for (uint32_t pass = 0; pass < 3; ++pass) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                uint32_t tile_id = batch_start_tile + ht * Wt + wt_in_batch;
                cb_reserve_back(cb_input, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_input);
                uint64_t noc_addr = input_accessor.get_noc_addr(tile_id);
                noc_async_read(noc_addr, l1_write_addr, page_size);
                noc_async_read_barrier();
                cb_push_back(cb_input, 1);
            }
        }
    }
#endif
}
