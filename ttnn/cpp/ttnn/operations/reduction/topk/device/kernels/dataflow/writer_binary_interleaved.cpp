// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t dst_addr0 = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr1 = get_arg_val<uint32_t>(1);
    const uint32_t id = get_arg_val<uint32_t>(2);
    const uint32_t work_per_core = get_arg_val<uint32_t>(3);

    // Compile time args
    constexpr uint32_t values_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(4);
    constexpr auto interleaved_accessor0_args = TensorAccessorArgs<5>();
    constexpr auto interleaved_accessor1_args =
        TensorAccessorArgs<interleaved_accessor0_args.next_compile_time_args_offset()>();

    // Constants
    constexpr uint32_t onetile = 1;

    // Tensor config
    const uint32_t tile_bytes_values = get_tile_size(values_cb_index);
    const auto interleaved_accessor0 = TensorAccessor(interleaved_accessor0_args, dst_addr0, tile_bytes_values);
    const uint32_t tile_bytes_ind = get_tile_size(output_ind_cb_index);
    const auto interleaved_accessor1 = TensorAccessor(interleaved_accessor1_args, dst_addr1, tile_bytes_ind);

    // Get Kt rows of values and then Kt rows of indices from compute kernel
    for (uint32_t core_loop = 0; core_loop < work_per_core; core_loop++) {
        const uint32_t j = id + core_loop * total_number_of_cores;
        // TopK values
        for (uint32_t i = 0; i < Kt; ++i) {
            cb_wait_front(values_cb_index, onetile);
            const uint32_t l1_read_addr = get_read_ptr(values_cb_index);
            noc_async_write_tile(j * Kt + i, interleaved_accessor0, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(values_cb_index, onetile);
        }  // i loop

        // TopK indices
        for (uint32_t i = 0; i < Kt; ++i) {
            cb_wait_front(output_ind_cb_index, onetile);
            const uint32_t l1_read_addr = get_read_ptr(output_ind_cb_index);
            noc_async_write_tile(j * Kt + i, interleaved_accessor1, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(output_ind_cb_index, onetile);
        }  // i loop
    }  // core_loop loop
}
