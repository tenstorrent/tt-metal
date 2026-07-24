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

// RM gather reader — column-distributed multi-core. Each core owns [w_start, w_start+w_per_core) of every row.
// Reads its index slice via noc_async_read_sharded (per-shard page size); writes gather result to output_cb.
void kernel_main() {
    // Runtime args
    const uint32_t input_index_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_per_core = get_arg_val<uint32_t>(1);
    const uint32_t w_start = get_arg_val<uint32_t>(2);
    const uint32_t core_id = get_arg_val<uint32_t>(3);
    const uint32_t input_index_per_shard_page_size_bytes = get_arg_val<uint32_t>(4);

    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t input_index_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t output_tensor_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t H = get_compile_time_arg_val(3);
    constexpr uint32_t W_input = get_compile_time_arg_val(4);
    constexpr uint32_t W_index = get_compile_time_arg_val(5);
    constexpr uint32_t input_tensor_data_format_size = get_compile_time_arg_val(6);
    constexpr uint32_t input_index_tensor_data_format_size = get_compile_time_arg_val(7);
    constexpr uint32_t output_tensor_data_format_size = get_compile_time_arg_val(8);
    constexpr auto input_index_tensor_args = TensorAccessorArgs<9>();

    // Idle cores (work_per_core == 0 from split_work_to_cores) skip without touching CBs.
    if (w_per_core == 0) {
        return;
    }

    constexpr uint32_t one_stick = 1;
    const uint32_t input_index_byte_offset = w_start * input_index_tensor_data_format_size;
    const uint32_t input_index_slice_bytes = w_per_core * input_index_tensor_data_format_size;

    const auto input_index_accessor =
        TensorAccessor(input_index_tensor_args, input_index_tensor_buffer_addr, input_index_per_shard_page_size_bytes);

    Noc noc;
    CircularBuffer input_cb(input_tensor_cb_index);
    CircularBuffer input_index_cb(input_index_tensor_cb_index);
    CircularBuffer output_cb(output_tensor_cb_index);

    for (uint32_t h = 0; h < H; h++) {
        input_index_cb.reserve_back(one_stick);
        tt::data_movement::common::noc_async_read_sharded(
            input_index_cb.get_write_ptr(),
            input_index_accessor,
            /*src_id=*/h,
            /*offset=*/input_index_byte_offset,
            /*size=*/input_index_slice_bytes);
        noc.async_read_barrier();
        input_index_cb.push_back(one_stick);

        input_cb.wait_front(one_stick);
        output_cb.reserve_back(one_stick);

        const uint32_t input_l1_read_addr = input_cb.get_read_ptr();
        const uint32_t input_index_l1_read_addr = input_index_cb.get_read_ptr();
        const uint32_t output_l1_write_addr = output_cb.get_write_ptr();

        // Linear gather over this core's slice: output[w] = input[index[w]] for
        // w in [0, w_per_core). The full input row is in input_cb so any index is valid.
        for (uint32_t w_local = 0; w_local < w_per_core; w_local++) {
            const uint32_t global_index =
                get_value_from_stick(input_index_l1_read_addr, w_local, input_index_tensor_data_format_size);
            const uint32_t value =
                get_value_from_stick(input_l1_read_addr, global_index, input_tensor_data_format_size);
            write_value_to_stick(output_l1_write_addr, w_local, output_tensor_data_format_size, value);
        }

        output_cb.push_back(one_stick);
        input_index_cb.pop_front(one_stick);
        input_cb.pop_front(one_stick);
    }
}
