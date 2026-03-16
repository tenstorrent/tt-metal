// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// CB indices
constexpr uint32_t c_16 = 16;  // output

// Compile-time args
constexpr uint32_t Ht = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t HtWt = get_compile_time_arg_val(2);
constexpr uint32_t dim = get_compile_time_arg_val(3);         // 0 = width (dim=-1), 1 = height (dim=-2)
constexpr auto output_tensor_args = TensorAccessorArgs<4>();  // Output tensor accessor

void kernel_main() {
    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_work_units = get_arg_val<uint32_t>(1);
    const uint32_t start_work_unit = get_arg_val<uint32_t>(2);

    constexpr uint32_t page_size = get_tile_size(c_16);
    const auto output_accessor = TensorAccessor(output_tensor_args, dst_addr, page_size);

    if constexpr (dim == 0) {
        // dim=-1 (width): work unit = tile-row
        for (uint32_t wu = 0; wu < num_work_units; wu++) {
            const uint32_t row_idx = start_work_unit + wu;
            const uint32_t nc = row_idx / Ht;
            const uint32_t ht = row_idx % Ht;
            const uint32_t base_tile_id = nc * HtWt + ht * Wt;

            // Write Wt tiles for this row
            for (uint32_t wt = 0; wt < Wt; wt++) {
                cb_wait_front(c_16, 1);
                uint32_t l1_addr = get_read_ptr(c_16);
                uint64_t noc_addr = output_accessor.get_noc_addr(base_tile_id + wt);
                noc_async_write(l1_addr, noc_addr, page_size);
                noc_async_write_barrier();
                cb_pop_front(c_16, 1);
            }
        }
    } else {
        // dim=-2 (height): work unit = tile-column
        for (uint32_t wu = 0; wu < num_work_units; wu++) {
            const uint32_t col_idx = start_work_unit + wu;
            const uint32_t nc = col_idx / Wt;
            const uint32_t wt = col_idx % Wt;
            const uint32_t base_tile_id = nc * HtWt;

            // Write Ht tiles for this column (strided access)
            for (uint32_t ht = 0; ht < Ht; ht++) {
                cb_wait_front(c_16, 1);
                uint32_t l1_addr = get_read_ptr(c_16);
                uint64_t noc_addr = output_accessor.get_noc_addr(base_tile_id + ht * Wt + wt);
                noc_async_write(l1_addr, noc_addr, page_size);
                noc_async_write_barrier();
                cb_pop_front(c_16, 1);
            }
        }
    }
}
