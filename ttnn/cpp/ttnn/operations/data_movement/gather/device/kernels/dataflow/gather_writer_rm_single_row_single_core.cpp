// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

#include <cstdint>

// RM gather writer: row-distributed single-core. Parks input rows in input_dfb; drains output_dfb to memory.
// I/O via noc_async_*_sharded with per-shard page sizes for B/W RM buffers.
void kernel_main() {
    // Runtime args
    const uint32_t core_loop_count = get_arg(args::core_loop_count);
    const uint32_t core_id = get_arg(args::core_id);

    // Compile time args
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t W_input = get_arg(args::W_input);
    constexpr uint32_t W_index = get_arg(args::W_index);
    constexpr uint32_t total_number_of_cores = get_arg(args::total_number_of_cores);
    constexpr uint32_t input_tensor_data_format_size = get_arg(args::input_tensor_data_format_size);
    constexpr uint32_t output_tensor_data_format_size = get_arg(args::output_tensor_data_format_size);

    constexpr uint32_t one_stick = 1;
    constexpr uint32_t input_stick_size_bytes = W_input * input_tensor_data_format_size;
    constexpr uint32_t output_stick_size_bytes = W_index * output_tensor_data_format_size;

    const auto input_accessor = TensorAccessor(tensor::input_tensor);
    const auto output_accessor = TensorAccessor(tensor::output_tensor);

    Noc noc;
    DataflowBuffer input_dfb(dfb::input_tensor);
    DataflowBuffer output_dfb(dfb::output_tensor);

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const uint32_t h = core_loop * total_number_of_cores + core_id;

        input_dfb.reserve_back(one_stick);
        tt::data_movement::common::noc_async_read_sharded(
            noc,
            input_dfb.get_write_ptr(),
            input_accessor,
            /*src_id=*/h,
            /*offset=*/0,
            /*size=*/input_stick_size_bytes);
        noc.async_read_barrier();
        input_dfb.push_back(one_stick);

        output_dfb.wait_front(one_stick);
        tt::data_movement::common::noc_async_write_sharded(
            noc,
            output_dfb.get_read_ptr(),
            output_accessor,
            /*dest_id=*/h,
            /*offset=*/0,
            /*size=*/output_stick_size_bytes);
        noc.async_write_barrier();
        output_dfb.pop_front(one_stick);
    }
}
