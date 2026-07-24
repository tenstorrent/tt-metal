// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

#include <cstdint>

// RM gather writer — row-distributed single-core. Parks input rows in input_cb; drains output_cb to memory.
// I/O via noc_async_*_sharded with per-shard page sizes for B/W RM buffers.
void kernel_main() {
    // Runtime args
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_tensor_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(2);
    const uint32_t core_id = get_arg_val<uint32_t>(3);
    const uint32_t input_per_shard_page_size_bytes = get_arg_val<uint32_t>(4);
    const uint32_t output_per_shard_page_size_bytes = get_arg_val<uint32_t>(5);

    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t W_input = get_compile_time_arg_val(3);
    constexpr uint32_t W_index = get_compile_time_arg_val(4);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(5);
    constexpr uint32_t input_tensor_data_format_size = get_compile_time_arg_val(6);
    constexpr uint32_t output_tensor_data_format_size = get_compile_time_arg_val(7);
    constexpr auto input_tensor_args = TensorAccessorArgs<8>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t one_stick = 1;
    constexpr uint32_t input_stick_size_bytes = W_input * input_tensor_data_format_size;
    constexpr uint32_t output_stick_size_bytes = W_index * output_tensor_data_format_size;

    const auto input_accessor =
        TensorAccessor(input_tensor_args, input_tensor_buffer_addr, input_per_shard_page_size_bytes);
    const auto output_accessor =
        TensorAccessor(output_tensor_args, output_tensor_buffer_addr, output_per_shard_page_size_bytes);

    Noc noc;
    CircularBuffer input_cb(input_tensor_cb_index);
    CircularBuffer output_cb(output_tensor_cb_index);

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const uint32_t h = core_loop * total_number_of_cores + core_id;

        input_cb.reserve_back(one_stick);
        tt::data_movement::common::noc_async_read_sharded(
            input_cb.get_write_ptr(),
            input_accessor,
            /*src_id=*/h,
            /*offset=*/0,
            /*size=*/input_stick_size_bytes);
        noc.async_read_barrier();
        input_cb.push_back(one_stick);

        output_cb.wait_front(one_stick);
        tt::data_movement::common::noc_async_write_sharded(
            output_cb.get_read_ptr(),
            output_accessor,
            /*dest_id=*/h,
            /*offset=*/0,
            /*size=*/output_stick_size_bytes);
        noc.async_write_barrier();
        output_cb.pop_front(one_stick);
    }
}
