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
    constexpr auto inout_tensor_args = TensorAccessorArgs<6>();

#if not GENERATE_INDICES
    // Precomputed indices tensor accessor
    constexpr auto indices_args = TensorAccessorArgs<inout_tensor_args.next_compile_time_args_offset()>();
    const uint32_t src_indices_addr = get_arg_val<uint32_t>(3);
    constexpr uint32_t indices_tile_bytes = get_tile_size(cb_intermed_index);
    const auto indices_accessor = TensorAccessor(indices_args, src_indices_addr, indices_tile_bytes);
#endif  // not GENERATE_INDICES

    // Constants
    constexpr uint32_t onetile = 1;

    // Tensor accessor
    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const auto inout_tensor_accessor = TensorAccessor(inout_tensor_args, src_addr, tile_bytes);

    // Read data and generate indices
    for (uint32_t core_loop = 0; core_loop < work_per_core; core_loop++) {
        const uint32_t row = id + core_loop * total_number_of_cores;
        for (uint32_t w = 0; w < Wt; ++w) {
            cb_reserve_back(cb_id_in0, onetile);
            const uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            noc_async_read_tile(row * Wt + w, inout_tensor_accessor, l1_write_addr);
            noc_async_read_barrier();
            DPRINT << "Reader: core_loop: " << core_loop << ", row: " << row << ", w: " << w
                   << " ,address: " << l1_write_addr << ENDL();
            // volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
            // for (int subtile_i = 0; subtile_i < 2; subtile_i++) {
            //     // Iterate through 16 rows within each subtile row
            //     for (int local_row = 0; local_row < 16; local_row++) {
            //         // Calculate the actual row in original matrix
            //         int row = subtile_i * 16 + local_row;
            //         // Iterate through 2x2 subtiles horizontally
            //         for (int subtile_j = 0; subtile_j < 2; subtile_j++) {
            //             // Iterate through 16 columns within each subtile
            //             for (int local_col = 0; local_col < 16; local_col++) {
            //                 // Calculate the actual column in original matrix
            //                 int col = subtile_j * 16 + local_col;
            //                 // Calculate index using only multiplication and addition
            //                 auto index = local_row * 16 + local_col + subtile_i * 512 + subtile_j * 256;
            //                 DPRINT << BF16(ptr[index]) << ", " ;
            //             }
            //         }
            //         DPRINT << ENDL();
            //     }
            // }
            // DPRINT << ENDL();
            cb_push_back(cb_id_in0, onetile);
#if GENERATE_INDICES
            if (uint16_output) {
                // DPRINT << "Reader: core_loop: " << core_loop << ", row: " << row << ", w: " << w << ENDL();
                generate_index_tile<uint16_t>(cb_intermed_index, w);
            } else {
                generate_index_tile<uint32_t>(cb_intermed_index, w);
            }
#else
            // Read precomputed indices to circular buffer
            cb_reserve_back(cb_intermed_index, onetile);
            const uint32_t l1_write_addr_ind = get_write_ptr(cb_intermed_index);
            noc_async_read_tile(row * Wt + w, indices_accessor, l1_write_addr_ind);
            noc_async_read_barrier();
            cb_push_back(cb_intermed_index, onetile);
#endif  // GENERATE_INDICES
        }  // w loop
        // Add delay loop with assembly NOPs
        // for (uint32_t nop_i = 0; nop_i < 10000000; ++nop_i) {
        //     asm volatile("nop");
        // }
    }  // core_loop loop
}
