// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

#include <cstdint>

// RM gather writer — column-distributed multi-core. Parks FULL input row in input_cb (reader needs any w).
// Drains the reader's output slice to memory at [w_start, w_start+w_per_core) via noc_async_*_sharded.
void kernel_main() {
    // Runtime args
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_tensor_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_per_core = get_arg_val<uint32_t>(2);
    const uint32_t w_start = get_arg_val<uint32_t>(3);
    const uint32_t core_id = get_arg_val<uint32_t>(4);
    const uint32_t input_per_shard_page_size_bytes = get_arg_val<uint32_t>(5);
    const uint32_t output_per_shard_page_size_bytes = get_arg_val<uint32_t>(6);

    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t W_input = get_compile_time_arg_val(3);
    constexpr uint32_t W_index = get_compile_time_arg_val(4);
    constexpr uint32_t input_tensor_data_format_size = get_compile_time_arg_val(5);
    constexpr uint32_t output_tensor_data_format_size = get_compile_time_arg_val(6);
    constexpr auto input_tensor_args = TensorAccessorArgs<7>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    if (w_per_core == 0) {
        return;
    }

    constexpr uint32_t one_stick = 1;
    constexpr uint32_t input_stick_size_bytes = W_input * input_tensor_data_format_size;
    const uint32_t output_byte_offset = w_start * output_tensor_data_format_size;
    const uint32_t output_slice_bytes = w_per_core * output_tensor_data_format_size;

    const auto input_accessor =
        TensorAccessor(input_tensor_args, input_tensor_buffer_addr, input_per_shard_page_size_bytes);
    const auto output_accessor =
        TensorAccessor(output_tensor_args, output_tensor_buffer_addr, output_per_shard_page_size_bytes);

    Noc noc;
    CircularBuffer input_cb(input_tensor_cb_index);
    CircularBuffer output_cb(output_tensor_cb_index);

    for (uint32_t h = 0; h < H; h++) {
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
            /*offset=*/output_byte_offset,
            /*size=*/output_slice_bytes);
        noc.async_write_barrier();
        output_cb.pop_front(one_stick);
    }
}
