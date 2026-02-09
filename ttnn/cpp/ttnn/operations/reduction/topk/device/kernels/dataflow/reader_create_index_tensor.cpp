// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_dataflow_common.hpp"

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

void kernel_main() {
    // Runtime arguments
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t id = get_arg_val<uint32_t>(1);
    const uint32_t work_per_core = get_arg_val<uint32_t>(2);

    // Compile time arguments
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_intermed_index = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(4);
    constexpr bool uint16_output = get_compile_time_arg_val(5) == 1;
    constexpr auto s_args = TensorAccessorArgs<6>();

#if not GENERATE_INDICES
    // Precomputed indices tensor accessor
    constexpr auto indices_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const uint32_t src_indices_addr = get_arg_val<uint32_t>(3);
    constexpr uint32_t indices_tile_bytes = get_tile_size(cb_intermed_index);
    const auto indices_accessor = TensorAccessor(indices_args, src_indices_addr, indices_tile_bytes);
#endif  // not GENERATE_INDICES

    // Constants
    constexpr uint32_t onetile = 1;

    // Tensor accessor
    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const auto s = TensorAccessor(s_args, src_addr, tile_bytes);

    // Read data and generate indices
    for (uint32_t core_loop = 0; core_loop < work_per_core; core_loop++) {
        const uint32_t i = id + core_loop * total_number_of_cores;
        for (uint32_t j = 0; j < Wt; ++j) {
            cb_reserve_back(cb_id_in0, onetile);
            const uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            noc_async_read_tile(i * Wt + j, s, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
#if GENERATE_INDICES
            if (uint16_output) {
                generate_index_tile<uint16_t>(cb_intermed_index, j);
            } else {
                generate_index_tile<uint32_t>(cb_intermed_index, j);
            }
#else
            // Read precomputed indices to circular buffer
            cb_reserve_back(cb_intermed_index, onetile);
            const uint32_t l1_write_addr_ind = get_write_ptr(cb_intermed_index);
            noc_async_read_tile(i * Wt + j, indices_accessor, l1_write_addr_ind);
            noc_async_read_barrier();
            cb_push_back(cb_intermed_index, onetile);
#endif  // GENERATE_INDICES
        }  // j loop
    }  // core_loop loop
}
