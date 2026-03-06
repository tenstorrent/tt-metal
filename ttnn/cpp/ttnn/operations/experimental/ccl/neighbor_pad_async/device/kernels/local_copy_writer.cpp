// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>
#include <utility>
//
using address_t = uint32_t;
//
constexpr uint32_t cb_output_id = get_compile_time_arg_val(0);
constexpr uint32_t stick_size = get_compile_time_arg_val(1);
//
void kernel_main() {
    // Args
    uint32_t arg_idx = 0;
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);  // not used in writer
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t total_rows_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stick_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rows_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding_left = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_per_halo_dim = get_arg_val<uint32_t>(arg_idx++);
    //

    constexpr auto dst_args = TensorAccessorArgs<2>();
    const auto dst_accessor = TensorAccessor(dst_args, output_tensor_address, stick_size);
    //
    for (uint32_t s = 0; s < rows_count; s++) {
        const uint32_t linear_row = total_rows_start + s;  // [0 .. outer_dim_size*input_halo_dim_size)
        const uint32_t outer_idx = linear_row / input_halo_dim_size;
        const uint32_t t = linear_row % input_halo_dim_size;
        const uint32_t outer_dim_offset = outer_idx * (num_sticks_per_halo_dim * output_halo_dim_size);

        uint32_t dst_stick_id = (t + padding_left) * num_sticks_per_halo_dim + stick_start_id + outer_dim_offset;
        for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
            cb_wait_front(cb_output_id, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_output_id);
            uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, dst_accessor);
            noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
            dst_stick_id++;
            noc_async_write_barrier();
            cb_pop_front(cb_output_id, 1);
        }
    }
    noc_async_write_barrier();
}
