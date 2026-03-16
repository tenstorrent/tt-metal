// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // --- Compile-time args ---
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);   // 0
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);  // 8
    constexpr uint32_t R = get_compile_time_arg_val(2);          // tiles per work unit
    constexpr uint32_t is_dim_h = get_compile_time_arg_val(3);   // 1 if dim=-2, 0 if dim=-1
    constexpr uint32_t Wt = get_compile_time_arg_val(4);         // width in tiles
    constexpr auto src_args = TensorAccessorArgs<5>();           // TensorAccessor CT args

    // --- Runtime args ---
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_work_units = get_arg_val<uint32_t>(1);
    const uint32_t start_work_unit = get_arg_val<uint32_t>(2);

    constexpr uint32_t page_size = get_tile_size(cb_input);
    const auto src = TensorAccessor(src_args, src_addr, page_size);

    // Prepare the reduce scaler tile (1.0f) once
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f);

    // Process each work unit
    for (uint32_t wu = 0; wu < num_work_units; ++wu) {
        uint32_t wu_idx = start_work_unit + wu;

        cb_reserve_back(cb_input, R);
        uint32_t l1_write_addr = get_write_ptr(cb_input);

        for (uint32_t t = 0; t < R; ++t) {
            uint32_t page_id;
            if constexpr (is_dim_h == 0) {
                // dim=-1: contiguous tiles
                page_id = wu_idx * R + t;
            } else {
                // dim=-2: strided access
                // wu_idx encodes (nc_idx * Wt + col_wt)
                uint32_t col_wt = wu_idx % Wt;
                uint32_t nc_ht_base = (wu_idx / Wt) * R * Wt;
                page_id = nc_ht_base + t * Wt + col_wt;
            }

            uint64_t noc_addr = src.get_noc_addr(page_id);
            noc_async_read(noc_addr, l1_write_addr, page_size);
            l1_write_addr += page_size;
        }

        noc_async_read_barrier();
        cb_push_back(cb_input, R);
    }
}
