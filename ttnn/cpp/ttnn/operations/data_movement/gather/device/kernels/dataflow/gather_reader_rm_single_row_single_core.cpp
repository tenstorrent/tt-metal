// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_common.hpp"

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

#include <cstdint>

// RM gather reader — row-distributed single-core. Reads index stick, gathers from input_cb, writes to output_cb.
// I/O via noc_async_*_sharded with per-shard page size; same kernel covers interleaved + H/B/W-sharded.
void kernel_main() {
    // Runtime args
    const uint32_t input_index_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(1);
    const uint32_t core_id = get_arg_val<uint32_t>(2);
    const uint32_t input_index_per_shard_page_size_bytes = get_arg_val<uint32_t>(3);

    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t input_index_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t output_tensor_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t H = get_compile_time_arg_val(3);
    constexpr uint32_t W_input = get_compile_time_arg_val(4);
    constexpr uint32_t W_index = get_compile_time_arg_val(5);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(6);
    constexpr uint32_t input_tensor_data_format_size = get_compile_time_arg_val(7);
    constexpr uint32_t input_index_tensor_data_format_size = get_compile_time_arg_val(8);
    constexpr uint32_t output_tensor_data_format_size = get_compile_time_arg_val(9);
    constexpr auto input_index_tensor_args = TensorAccessorArgs<10>();

    constexpr uint32_t one_stick = 1;
    constexpr uint32_t input_index_stick_size_bytes = W_index * input_index_tensor_data_format_size;

    // Per-shard page size drives the multi-shard split: shard_W * elem_size for B/W, full row otherwise.
    const auto input_index_accessor =
        TensorAccessor(input_index_tensor_args, input_index_tensor_buffer_addr, input_index_per_shard_page_size_bytes);

    Noc noc;
    CircularBuffer input_cb(input_tensor_cb_index);
    CircularBuffer input_index_cb(input_index_tensor_cb_index);
    CircularBuffer output_cb(output_tensor_cb_index);

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const uint32_t h = core_loop * total_number_of_cores + core_id;

        input_index_cb.reserve_back(one_stick);
        tt::data_movement::common::noc_async_read_sharded(
            input_index_cb.get_write_ptr(),
            input_index_accessor,
            /*src_id=*/h,
            /*offset=*/0,
            /*size=*/input_index_stick_size_bytes);
        noc.async_read_barrier();
        input_index_cb.push_back(one_stick);

        input_cb.wait_front(one_stick);
        output_cb.reserve_back(one_stick);

        const uint32_t input_l1_read_addr = input_cb.get_read_ptr();
        const uint32_t input_index_l1_read_addr = input_index_cb.get_read_ptr();
        const uint32_t output_l1_write_addr = output_cb.get_write_ptr();

        // Linear gather: output[w] = input[index[w]]. No tile-face math — RM data is
        // contiguous and `get_value_from_stick` indexes by element offset directly.
        for (uint32_t w = 0; w < W_index; w++) {
            const uint32_t global_index =
                get_value_from_stick(input_index_l1_read_addr, w, input_index_tensor_data_format_size);
            const uint32_t value =
                get_value_from_stick(input_l1_read_addr, global_index, input_tensor_data_format_size);
            write_value_to_stick(output_l1_write_addr, w, output_tensor_data_format_size, value);
        }

        output_cb.push_back(one_stick);
        input_index_cb.pop_front(one_stick);
        input_cb.pop_front(one_stick);
    }
}
