// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // --- Compile-time args ---
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);    // 16
    constexpr uint32_t R = get_compile_time_arg_val(1);         // tiles per work unit
    constexpr uint32_t is_dim_h = get_compile_time_arg_val(2);  // 1 if dim=-2, 0 if dim=-1
    constexpr uint32_t Wt = get_compile_time_arg_val(3);        // width in tiles
    constexpr auto dst_args = TensorAccessorArgs<4>();          // TensorAccessor CT args

    // --- Runtime args ---
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_work_units = get_arg_val<uint32_t>(1);
    const uint32_t start_work_unit = get_arg_val<uint32_t>(2);

    constexpr uint32_t page_size = get_tile_size(cb_out);
    const auto dst = TensorAccessor(dst_args, dst_addr, page_size);

    for (uint32_t wu = 0; wu < num_work_units; ++wu) {
        uint32_t wu_idx = start_work_unit + wu;

        for (uint32_t t = 0; t < R; ++t) {
            // Wait for 1 tile at a time (cb_out is double-buffered with 2 pages)
            cb_wait_front(cb_out, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_out);

            uint32_t page_id;
            if constexpr (is_dim_h == 0) {
                // dim=-1: contiguous tiles
                page_id = wu_idx * R + t;
            } else {
                // dim=-2: strided access
                uint32_t col_wt = wu_idx % Wt;
                uint32_t nc_ht_base = (wu_idx / Wt) * R * Wt;
                page_id = nc_ht_base + t * Wt + col_wt;
            }

            uint64_t noc_addr = dst.get_noc_addr(page_id);
            noc_async_write(l1_read_addr, noc_addr, page_size);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
