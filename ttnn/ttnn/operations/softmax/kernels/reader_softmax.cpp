// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// CB indices
constexpr uint32_t c_0 = 0;  // input
constexpr uint32_t c_1 = 1;  // scaler

// Compile-time args
constexpr uint32_t Ht = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t HtWt = get_compile_time_arg_val(2);
constexpr uint32_t dim = get_compile_time_arg_val(3);  // 0 = width (dim=-1), 1 = height (dim=-2)
constexpr uint32_t numeric_stable = get_compile_time_arg_val(4);
constexpr auto input_tensor_args = TensorAccessorArgs<5>();  // Input tensor accessor

void kernel_main() {
    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_work_units = get_arg_val<uint32_t>(1);
    const uint32_t start_work_unit = get_arg_val<uint32_t>(2);

    constexpr uint32_t page_size = get_tile_size(c_0);
    const auto input_accessor = TensorAccessor(input_tensor_args, src_addr, page_size);

    // Generate scaler tile in c_1 (all 1.0 for reduce)
    dataflow_kernel_lib::prepare_reduce_scaler<c_1>(1.0f);

    if constexpr (dim == 0) {
        // dim=-1 (width): work unit = tile-row
        // Each work unit is one tile-row of Wt tiles
        // For passthrough (stage 1): single pass, stream all tiles
        for (uint32_t wu = 0; wu < num_work_units; wu++) {
            const uint32_t row_idx = start_work_unit + wu;
            // row_idx = nc * Ht + ht
            const uint32_t nc = row_idx / Ht;
            const uint32_t ht = row_idx % Ht;
            const uint32_t base_tile_id = nc * HtWt + ht * Wt;

            // Stream Wt tiles for this row
            for (uint32_t wt = 0; wt < Wt; wt++) {
                cb_reserve_back(c_0, 1);
                uint32_t l1_addr = get_write_ptr(c_0);
                uint64_t noc_addr = input_accessor.get_noc_addr(base_tile_id + wt);
                noc_async_read(noc_addr, l1_addr, page_size);
                noc_async_read_barrier();
                cb_push_back(c_0, 1);
            }
        }
    } else {
        // dim=-2 (height): work unit = tile-column
        // Each work unit is one tile-column of Ht tiles
        for (uint32_t wu = 0; wu < num_work_units; wu++) {
            const uint32_t col_idx = start_work_unit + wu;
            // col_idx = nc * Wt + wt
            const uint32_t nc = col_idx / Wt;
            const uint32_t wt = col_idx % Wt;
            const uint32_t base_tile_id = nc * HtWt;

            // Stream Ht tiles for this column (strided access)
            for (uint32_t ht = 0; ht < Ht; ht++) {
                cb_reserve_back(c_0, 1);
                uint32_t l1_addr = get_write_ptr(c_0);
                uint64_t noc_addr = input_accessor.get_noc_addr(base_tile_id + ht * Wt + wt);
                noc_async_read(noc_addr, l1_addr, page_size);
                noc_async_read_barrier();
                cb_push_back(c_0, 1);
            }
        }
    }
}
